import torch
import julius
from . import EncodecModel, DiffusionModel
import typing as tp
import omegaconf
import os


def get_processor(cfg, sample_rate: int = 24000):
    if cfg.use:
        kw = dict(cfg)
        kw.pop('use')
        kw.pop('name')
        sample_processor = MultiBandProcessor(sample_rate=sample_rate, **kw)
    else:
        sample_processor = SampleProcessor()
    return sample_processor


class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor, ref_audio: tp.Optional[torch.Tensor] = None):
        """Project the original sample to the 'space' where
        the diffusion will happen."""
        return x

    def return_sample(self, z: torch.Tensor, ref_audio: tp.Optional[torch.Tensor] = None):
        """Project back from diffusion space to the actual sample space.
        """
        return z


class MultiBandProcessor(SampleProcessor):
    """
    MultiBand sample processor. The input audio is splitted across
    frequency bands evenly distributed in mel-scale.

    Each band will be rescaled to match the power distribution
    of Gaussian noise in that band, using online metrics
    computed on the first few samples.

    Args:
        n_bands: number of mel-bands to split the signal over.
        sample_rate: sample rate of the audio.
        num_samples: number of samples to use to fit the rescaling
            for each band. The processor won't be stable
            until it has seen that many samples.
        power_std: The rescaling factor computed to match the
            power of Gaussian noise in each band is taken to
            that power, i.e. `1.` means full correction of the energy
            in each band, and values less than `1` means only partial
            correction. Can be used to balance the relative importance
            of low vs. high freq in typical audio signals.
    """
    def __init__(self, n_bands: int = 8, sample_rate: float = 24_000,
                 num_samples: int = 10_000, power_std: tp.Union[float, list[float]] = 1.):
        super().__init__()
        self.n_bands = n_bands
        self.split_bands = julius.SplitBands(sample_rate, n_bands=n_bands)
        self.num_samples = num_samples
        self.power_std = power_std
        if type(power_std) == list:
            assert len(power_std) == n_bands
            power_std = torch.tensor(power_std)
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(n_bands))
        self.register_buffer('sum_x2', torch.zeros(n_bands))
        self.register_buffer('sum_target_x2', torch.zeros(n_bands))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor
        self.sum_target_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return std

    @property
    def target_std(self):
        target_std = self.sum_target_x2 / self.counts
        return target_std

    def project_sample(self, x: torch.Tensor, ref_audio: tp.Optional[torch.Tensor] = None):
        assert x.dim() == 3
        bands = self.split_bands(x)
        if self.counts.item() < self.num_samples:
            ref_bands = self.split_bands(torch.randn_like(x))
            self.counts += len(x)
            self.sum_x += bands.mean(dim=(2, 3)).sum(dim=1)
            self.sum_x2 += bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
            self.sum_target_x2 += ref_bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        bands = (bands - self.mean.view(-1, 1, 1, 1)) * rescale.view(-1, 1, 1, 1)
        return bands.sum(dim=0)

    def return_sample(self, x: torch.Tensor, ref_audio: tp.Optional[torch.Tensor] = None):
        assert x.dim() == 3
        bands = self.split_bands(x)
        rescale = (self.std / self.target_std) ** self.power_std
        bands = bands * rescale.view(-1, 1, 1, 1) + self.mean.view(-1, 1, 1, 1)
        return bands.sum(dim=0)


def betas_from_alpha_bar(alpha_bar: torch.Tensor):
    alphas = torch.cat([torch.Tensor([alpha_bar[0]]), alpha_bar[1:]/alpha_bar[:-1]])
    return 1 - alphas


class DiffusionProcess:
    """Sampling for a diffusion Model"""
    def __init__(self, model: DiffusionModel, sample_processor: SampleProcessor, device, num_steps: int, beta_t0: float,
                 beta_t1: float, beta_exp: float) -> None:
        self.model = model
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_t0 ** (1 / beta_exp), beta_t1 ** (1 / beta_exp), num_steps,
                                    device=device, dtype=torch.float) ** beta_exp
        self.sample_processor = sample_processor

    def get_alpha_bar(self, step: tp.Optional[tp.Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """Return 'alpha_bar', either for a given step, or as a tensor
        with its value for each step.
        """
        if step is None:
            return (1 - self.betas).cumprod(dim=-1)  # works for simgle and multi bands
        if type(step) is int:
            return (1 - self.betas[:step + 1]).prod()
        else:
            return (1 - self.betas).cumprod(dim=0)[step].view(-1, 1, 1)

    @torch.no_grad()
    def generate_subsampled(self, condition: torch.Tensor, step_list: tp.Optional[list] = None,
                            initial: tp.Optional[torch.Tensor] = None, return_list: bool = False,):
        if step_list is None:
            step_list = list(range(1000))[::-50] + [0]
        alpha_bar = self.get_alpha_bar(step=self.num_steps - 1)
        alpha_bars_subsampled = (1 - self.betas).cumprod(dim=0)[list(reversed(step_list))].cpu()
        betas_subsampled = betas_from_alpha_bar(alpha_bars_subsampled)
        current = initial
        for idx, step in enumerate(step_list[:-1]):
            with torch.no_grad():
                estimate = self.model(current, step, condition=condition).sample
            alpha = 1 - betas_subsampled[-1 - idx]
            previous = (current - (1 - alpha) / (1 - alpha_bar).sqrt() * estimate) / alpha.sqrt()
            previous_alpha_bar = self.get_alpha_bar(step_list[idx + 1])
            if step == step_list[-2]:
                sigma2 = 0
                previous_alpha_bar = 1.0
            else:
                sigma2 = (1 - previous_alpha_bar) / (1 - alpha_bar) * (1 - alpha)
            if sigma2 > 0:
                previous += sigma2**0.5 * torch.randn_like(previous)
            current = previous
            alpha_bar = previous_alpha_bar
            if step == 0:
                previous *= self.rescale
        return self.sample_processor.return_sample(previous)


class MultiBandDiffusion:
    """sample from multiple diffusion models"""
    def __init__(self, DPs, codec_model) -> None:
        self.DPs = DPs
        self.codec_model = codec_model
        self.device = next(self.codec_model.parameters()).device

    @staticmethod
    def get_mbd_24khz(bw: float = 3.0, pretrained: bool = True, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert bw in [1.5, 3.0, 6.0], "bandwidth not available"
        dirname = os.path.dirname(__file__)
        path = {1.5: os.path.join(dirname, "diffusion_configs/1.5kbps.yaml"),
                3.0: os.path.join(dirname, "diffusion_configs/3kbps.yaml"),
                6.0: os.path.join(dirname, "diffusion_configs/3kbps.yaml")}[bw]
        codec_model = EncodecModel.encodec_model_24khz()
        codec_model.set_target_bandwidth(bw)
        codec_model = codec_model.to(device)
        cfg = omegaconf.OmegaConf.load(path)
        DPs = []
        for i in range(cfg.n_bands):
            band_cfg = getattr(cfg, f'band_{i}')
            model = DiffusionModel(chin=1, **band_cfg.myunet)
            processor = get_processor(band_cfg.processor)
            if pretrained:
                model_state = torch.hub.load_state_dict_from_url(band_cfg.model_url, map_location=device,
                                                                 check_hash=True)
                model.load_state_dict(model_state)
                processor_state = torch.hub.load_state_dict_from_url(band_cfg.processor_url, map_location=device,
                                                                     check_hash=True)
                processor.load_state_dict(processor_state)
            model = model.to(device)
            processor = processor.to(device)
            DPs.append(DiffusionProcess(model=model, sample_processor=processor, device=device, **band_cfg.diffusion))
        return MultiBandDiffusion(DPs, codec_model)

    @torch.no_grad()
    def get_condition(self, wav: torch.Tensor, no_resampling=False) -> torch.Tensor:
        assert self.codec_model is not None
        quantized, _ = self.codec_model._encode_frame(wav.to(self.device))
        quantized = quantized.transpose(0, 1)
        emb = self.codec_model.quantizer.decode(quantized)
        return emb

    def generate(self, emb: torch.Tensor, size: tp.Optional[torch.Size] = None, step_list=None):
        if size is None:
            size = torch.Size([emb.size(0), 1, emb.size(-1) * 320])
        assert size[0] == emb.size(0)
        out = torch.zeros(size).to(self.device)
        for DP in self.DPs:
            out += DP.generate_subsampled(condition=emb, step_list=step_list, initial=torch.randn_like(out))
        return out

    def regenerate(self, wav: torch.Tensor):
        emb = self.get_condition(wav)
        size = wav.size()
        return self.generate(emb, size=size)


if __name__ == '__main__':
    import soundfile as sf
    wav, sr = sf.read('../out_test_24k.wav')
    wav_torch = torch.from_numpy(wav).float().cuda().view(1, 1, -1)
    Generator = MultiBandDiffusion()
    out = Generator.regenerate(wav_torch)
    sf.write('../out_test_24k.wav', out.cpu().numpy().squeeze(0).squeeze(0), samplerate=sr)
