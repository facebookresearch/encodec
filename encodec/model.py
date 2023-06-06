# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn

from . import quantization as qt
from . import modules as m
from .utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class Output:
    sample: torch.Tensor


ROOT_URL = 'https://dl.fbaipublicfiles.com/encodec/v0/'

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


class LMModel(nn.Module):
    """Language Model to estimate probabilities of each codebook entry.
    We predict all codebooks in parallel for a given time step.

    Args:
        n_q (int): number of codebooks.
        card (int): codebook cardinality.
        dim (int): transformer dimension.
        **kwargs: passed to `encodec.modules.transformer.StreamingTransformerEncoder`.
    """
    def __init__(self, n_q: int = 32, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        self.card = card
        self.n_q = n_q
        self.dim = dim
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices: torch.Tensor,
                states: tp.Optional[tp.List[torch.Tensor]] = None, offset: int = 0):
        """
        Args:
            indices (torch.Tensor): indices from the previous time step. Indices
                should be 1 + actual index in the codebook. The value 0 is reserved for
                when the index is missing (i.e. first time step). Shape should be
                `[B, n_q, T]`.
            states: state for the streaming decoding.
            offset: offset of the current time step.

        Returns a 3-tuple `(probabilities, new_states, new_offset)` with probabilities
        with a shape `[B, card, n_q, T]`.

        """
        B, K, T = indices.shape
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self,
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float],
                 sample_rate: int,
                 channels: int,
                 normalize: bool = False,
                 segment: tp.Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'unset'):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        return self.decode(frames)[:, :, :x.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth

    def get_lm_model(self) -> LMModel:
        """Return the associated LM model to improve the compression rate.
        """
        device = next(self.parameters()).device
        lm = LMModel(self.quantizer.n_q, self.quantizer.bins, num_layers=5, dim=200,
                     past_context=int(3.5 * self.frame_rate)).to(device)
        checkpoints = {
            'encodec_24khz': 'encodec_lm_24khz-1608e3c0.th',
            'encodec_48khz': 'encodec_lm_48khz-7add9fc3.th',
        }
        try:
            checkpoint_name = checkpoints[self.name]
        except KeyError:
            raise RuntimeError("No LM pre-trained for the current Encodec model.")
        url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
        state = torch.hub.load_state_dict_from_url(
            url, map_location='cpu', check_hash=True)  # type: ignore
        lm.load_state_dict(state)
        lm.eval()
        return lm

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = False,
                   segment: tp.Optional[float] = None,
                   name: str = 'unset'):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: tp.Optional[Path] = None):
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split('-')[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)  # type:ignore

    @staticmethod
    def encodec_model_24khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained causal 24khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        checkpoint_name = 'encodec_24khz-d7cc33bc.th'
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='encodec_24khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained 48khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [3., 6., 12., 24.]
        checkpoint_name = 'encodec_48khz-7e698e3e.th'
        sample_rate = 48_000
        channels = 2
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=False, model_norm='time_group_norm', audio_normalize=True,
            segment=1., name='encodec_48khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    """
    def __init__(self, dim, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, norm_groups: int = 4,
                 dilation: int = 1, activation: tp.Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.):
        super().__init__()
        stride = 1
        padding = dilation * (kernel - stride) // 2
        Conv = nn.Conv1d
        Drop = nn.Dropout1d
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv1 = Conv(channels, channels, kernel, 1, padding, dilation=dilation)
        self.activation1 = activation()
        self.dropout1 = Drop(dropout)

        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.conv2 = Conv(channels, channels, kernel, 1, padding, dilation=dilation)
        self.activation2 = activation()
        self.dropout2 = Drop(dropout)

    def forward(self, x):
        h = self.dropout1(self.conv1(self.activation1(self.norm1(x))))
        h = self.dropout2(self.conv2(self.activation2(self.norm2(h))))
        return x + h


class DecoderLayer(nn.Module):
    def __init__(self, chin: int, chout: int, kernel: int = 4, stride: int = 2,
                 norm_groups: int = 4, res_blocks: int = 1, activation: tp.Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.):
        super().__init__()
        padding = (kernel - stride) // 2
        self.res_blocks = nn.Sequential(
            *[ResBlock(chin, norm_groups=norm_groups, dilation=2**idx, dropout=dropout)
              for idx in range(res_blocks)])
        self.norm = nn.GroupNorm(norm_groups, chin)
        ConvTr = nn.ConvTranspose1d
        self.convtr = ConvTr(chin, chout, kernel, stride, padding, bias=False)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_blocks(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.convtr(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, chin: int, chout: int, kernel: int = 4, stride: int = 2,
                 norm_groups: int = 4, res_blocks: int = 1, activation: tp.Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.):
        super().__init__()
        padding = (kernel - stride) // 2
        Conv = nn.Conv1d
        self.conv = Conv(chin, chout, kernel, stride, padding, bias=False)
        self.norm = nn.GroupNorm(norm_groups, chout)
        self.activation = activation()
        self.res_blocks = nn.Sequential(
            *[ResBlock(chout, norm_groups=norm_groups, dilation=2**idx, dropout=dropout)
              for idx in range(res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, T = x.shape
        stride, = self.conv.stride
        pad = (stride - (T % stride)) % stride
        x = F.pad(x, (0, pad))

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.res_blocks(x)
        return x


class DiffusionModel(nn.Module):
    """Diffusion model. Diffusion model used in From Disrete Tokens to High Fidelity \
        Audio with Multi Band Diffusion Models"""
    def __init__(self, chin: int = 3, hidden: int = 24, depth: int = 3, growth: float = 2.,
                 max_channels: int = 10_000, num_steps: int = 1000, emb_all_layers=False, bilstm: bool = False,
                 codec_dim: tp.Optional[int] = None, **kwargs):
        super().__init__()
        init_hidden = hidden
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.embeddings: tp.Optional[nn.ModuleList] = None
        self.embedding = nn.Embedding(num_steps, hidden)
        if emb_all_layers:
            self.embeddings = nn.ModuleList()
        self.condition_embedding: tp.Optional[nn.Module] = None
        for d in range(depth):
            encoder = EncoderLayer(chin, hidden, **kwargs)
            decoder = DecoderLayer(hidden, chin, **kwargs)
            self.encoders.append(encoder)
            self.decoders.insert(0, decoder)
            if emb_all_layers and d > 0:
                assert self.embeddings is not None
                self.embeddings.append(nn.Embedding(num_steps, hidden))
            chin = hidden
            hidden = min(int(chin * growth), max_channels)
        self.bilstm: tp.Optional[nn.Module]
        if bilstm:
            self.bilstm = BLSTM(chin)
        else:
            self.bilstm = None
        self.conv_codec = nn.Conv1d(codec_dim, chin, 1)

    def forward(self, x: torch.Tensor, step: tp.Union[int, torch.Tensor], condition: tp.Optional[torch.Tensor] = None):
        skips = []
        bs = x.size(0)
        z = x
        view_args = [1]
        if type(step) is int:
            step_tensor = torch.tensor([step], device=x.device, dtype=torch.long).expand(bs)
        else:
            step_tensor = step
        for idx, encoder in enumerate(self.encoders):
            z = encoder(z)
            if idx == 0:
                z = z + self.embedding(step_tensor).view(bs, -1, *view_args).expand_as(z)
            elif self.embeddings is not None:
                z = z + self.embeddings[idx - 1](step_tensor).view(bs, -1, *view_args).expand_as(z)

            skips.append(z)

        assert condition is not None, "Model defined for conditionnal generation"

        condition_emb = self.conv_codec(condition)  # reshape to the bottleneck dim
        assert condition_emb.size(-1) <= 2 * z.size(-1), f"You are downsampling the conditionning with factor of at least 2 : {condition_emb.size(-1)=} and {z.size(-1)=}"
        condition_emb = torch.nn.functional.interpolate(condition_emb, z.size(-1))
        assert z.size() == condition_emb.size()
        z += condition_emb

        if self.bilstm is None:
            z = torch.zeros_like(z)
        else:
            z = self.bilstm(z)

        for decoder in self.decoders:
            s = skips.pop(-1)
            z = z[:, :, :s.shape[2]]
            z = z + s
            z = decoder(z)
        z = z[:, :, :x.shape[2]]
        return Output(z)


def test():
    from itertools import product
    import torchaudio
    bandwidths = [3, 6, 12, 24]
    models = {
        'encodec_24khz': EncodecModel.encodec_model_24khz,
        'encodec_48khz': EncodecModel.encodec_model_48khz
    }
    for model_name, bw in product(models.keys(), bandwidths):
        model = models[model_name]()
        model.set_target_bandwidth(bw)
        audio_suffix = model_name.split('_')[1][:3]
        wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")
        wav = wav[:, :model.sample_rate * 2]
        wav_in = wav.unsqueeze(0)
        wav_dec = model(wav_in)[0]
        assert wav.shape == wav_dec.shape, (wav.shape, wav_dec.shape)


if __name__ == '__main__':
    test()
