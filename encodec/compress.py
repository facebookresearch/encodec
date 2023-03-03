# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API to compress/decompress audio to bytestreams."""

import io
import math
import struct
import time
import typing as tp

import torch

from . import binary
from .quantization.ac import ArithmeticCoder, ArithmeticDecoder, build_stable_quantized_cdf
from .model import EncodecModel, EncodedFrame


MODELS = {
    'encodec_24khz': EncodecModel.encodec_model_24khz,
    'encodec_48khz': EncodecModel.encodec_model_48khz,
}


def compress_to_file(model: EncodecModel, wav: torch.Tensor, fo: tp.IO[bytes],
                     use_lm: bool = True):
    """Compress a waveform to a file-object using the given model.

    Args:
        model (EncodecModel): a pre-trained EncodecModel to use to compress the audio.
        wav (torch.Tensor): waveform to compress, should have a shape `[C, T]`, with `C`
            matching `model.channels`, and the proper sample rate (e.g. `model.sample_rate`).
            Use `utils.convert_audio` if this is not the case.
        fo (IO[bytes]): file-object to which the compressed bits will be written.
            See `compress` if you want obtain a `bytes` object instead.
        use_lm (bool): if True, use a pre-trained language model to further
            compress the stream using Entropy Coding. This will slow down compression
            quite a bit, expect between 20 to 30% of size reduction.
    """
    assert wav.dim() == 2, "Only single waveform can be encoded."
    if model.name not in MODELS:
        raise ValueError(f"The provided model {model.name} is not supported.")

    if use_lm:
        lm = model.get_lm_model()

    with torch.no_grad():
        frames = model.encode(wav[None])

    metadata = {
        'm': model.name,                 # model name
        'al': wav.shape[-1],             # audio_length
        'nc': frames[0][0].shape[1],     # num_codebooks
        'lm': use_lm,                    # use lm?
    }
    binary.write_ecdc_header(fo, metadata)

    for (frame, scale) in frames:
        if scale is not None:
            fo.write(struct.pack('!f', scale.cpu().item()))
        _, K, T = frame.shape
        if use_lm:
            coder = ArithmeticCoder(fo)
            states: tp.Any = None
            offset = 0
            input_ = torch.zeros(1, K, 1, dtype=torch.long, device=wav.device)
        else:
            packer = binary.BitPacker(model.bits_per_codebook, fo)
        for t in range(T):
            if use_lm:
                with torch.no_grad():
                    probas, states, offset = lm(input_, states, offset)
                # We emulate a streaming scenario even though we do not provide an API for it.
                # This gives us a more accurate benchmark.
                input_ = 1 + frame[:, :, t: t + 1]
            for k, value in enumerate(frame[0, :, t].tolist()):
                if use_lm:
                    q_cdf = build_stable_quantized_cdf(
                        probas[0, :, k, 0], coder.total_range_bits, check=False)
                    coder.push(value, q_cdf)
                else:
                    packer.push(value)
        if use_lm:
            coder.flush()
        else:
            packer.flush()


def decompress_from_file(fo: tp.IO[bytes], device='cpu') -> tp.Tuple[torch.Tensor, int]:
    """Decompress from a file-object.
    Returns a tuple `(wav, sample_rate)`.

    Args:
        fo (IO[bytes]): file-object from which to read. If you want to decompress
            from `bytes` instead, see `decompress`.
        device: device to use to perform the computations.
    """
    metadata = binary.read_ecdc_header(fo)
    model_name = metadata['m']
    audio_length = metadata['al']
    num_codebooks = metadata['nc']
    use_lm = metadata['lm']
    assert isinstance(audio_length, int)
    assert isinstance(num_codebooks, int)
    if model_name not in MODELS:
        raise ValueError(f"The audio was compressed with an unsupported model {model_name}.")
    model = MODELS[model_name]().to(device)

    if use_lm:
        lm = model.get_lm_model()

    frames: tp.List[EncodedFrame] = []
    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length
    for offset in range(0, audio_length, segment_stride):
        this_segment_length = min(audio_length - offset, segment_length)
        frame_length = int(math.ceil(this_segment_length * model.frame_rate / model.sample_rate))
        if model.normalize:
            scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
            scale = torch.tensor(scale_f, device=device).view(1)
        else:
            scale = None
        if use_lm:
            decoder = ArithmeticDecoder(fo)
            states: tp.Any = None
            offset = 0
            input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=device)
        else:
            unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)
        frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=device)
        for t in range(frame_length):
            if use_lm:
                with torch.no_grad():
                    probas, states, offset = lm(input_, states, offset)
            code_list: tp.List[int] = []
            for k in range(num_codebooks):
                if use_lm:
                    q_cdf = build_stable_quantized_cdf(
                        probas[0, :, k, 0], decoder.total_range_bits, check=False)
                    code = decoder.pull(q_cdf)
                else:
                    code = unpacker.pull()
                if code is None:
                    raise EOFError("The stream ended sooner than expected.")
                code_list.append(code)
            codes = torch.tensor(code_list, dtype=torch.long, device=device)
            frame[0, :, t] = codes
            if use_lm:
                input_ = 1 + frame[:, :, t: t + 1]
        frames.append((frame, scale))
    with torch.no_grad():
        wav = model.decode(frames)
    return wav[0, :, :audio_length], model.sample_rate


def compress(model: EncodecModel, wav: torch.Tensor, use_lm: bool = False) -> bytes:
    """Compress a waveform using the given model. Returns the compressed bytes.

    Args:
        model (EncodecModel): a pre-trained EncodecModel to use to compress the audio.
        wav (torch.Tensor): waveform to compress, should have a shape `[C, T]`, with `C`
            matching `model.channels`, and the proper sample rate (e.g. `model.sample_rate`).
            Use `utils.convert_audio` if this is not the case.
        use_lm (bool): if True, use a pre-trained language model to further
            compress the stream using Entropy Coding. This will slow down compression
            quite a bit, expect between 20 to 30% of size reduction.
    """
    fo = io.BytesIO()
    compress_to_file(model, wav, fo, use_lm=use_lm)
    return fo.getvalue()


def decompress(compressed: bytes, device='cpu') -> tp.Tuple[torch.Tensor, int]:
    """Decompress from a file-object.
    Returns a tuple `(wav, sample_rate)`.

    Args:
        compressed (bytes): compressed bytes.
        device: device to use to perform the computations.
    """
    fo = io.BytesIO(compressed)
    return decompress_from_file(fo, device=device)


def test():
    import torchaudio
    torch.set_num_threads(1)
    for name in MODELS.keys():
        model = MODELS[name]()
        sr = model.sample_rate // 1000
        x, _ = torchaudio.load(f'test_{sr}k.wav')
        x = x[:, :model.sample_rate * 5]
        model.set_target_bandwidth(12)
        for use_lm in [False, True]:
            print(f"Doing {name}, use_lm={use_lm}")
            begin = time.time()
            res = compress(model, x, use_lm=use_lm)
            t_comp = time.time() - begin
            x_dec, _ = decompress(res)
            t_decomp = time.time() - begin - t_comp
            kbps = 8 * len(res) / 1000 / (x.shape[-1] / model.sample_rate)
            print(f"kbps: {kbps:.1f}, time comp: {t_comp:.1f} sec. "
                  f"time decomp:{t_decomp:.1f}.")
            assert x_dec.shape == x.shape


if __name__ == '__main__':
    test()
