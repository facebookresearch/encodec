# EnCodec: High Fidelity Neural Audio Compression
![linter badge](https://github.com/facebookresearch/encodec/workflows/linter/badge.svg)
![tests badge](https://github.com/facebookresearch/encodec/workflows/tests/badge.svg)

This is the code for the EnCodec neural codec presented in the [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) [[abs]](https://arxiv.org/abs/2210.13438).
paper. We provide our two multi-bandwidth models:
* A causal model operating at 24 kHz on monophonic audio trained on a variety of audio data.
* A non-causal model operationg at 48 kHz on stereophonic audio trained on music-only data.

The 24 kHz model can compress to 1.5, 3, 6, 12 or 24 kbps, while the 48 kHz model
support 3, 6, 12 and 24 kbps. We also provide a pre-trained language model for each
of the models, that can further compress the representation by up to 40% without
any further loss of quality.

For reference, we also provide the code for our novel MS-STFT discriminator.

<p align="center">
<img src="./architecture.png" alt="Schema representing the structure of Encodec,
    with a convolutional+LSTM encoder, a Residual Vector Quantization in the middle,
    followed by a convolutional+LSTM decoder. A multiscale complex spectrogram discriminator is applied to the output, along with objective reconstruction losses.
    A small transformer model is trained to predict the RVQ output."
width="800px"></p>


## Samples

Samples including baselines are provided on [our sample page](https://ai.honu.io/papers/encodec/samples.html).
You can also have a quick demo of what we achieve for 48 kHz music with EnCodec, along with
entropy coding, by clicking the thumbnail (original tracks provided by [Lucille Crew](https://open.spotify.com/artist/5eLv7rNfrf3IjMnK311ByP?si=X_zD9ackRRGjFP5Y6Q7Zng) and [Voyageur I](https://open.spotify.com/artist/21HymveeIhDcM4KDKeNLz0?si=4zXF8VpeQpeKR9QUIuck9Q)).

<p align="center">
<a href="https://ai.honu.io/papers/encodec/final.mp4">
<img src="./thumbnail.png" alt="Thumbnail for the sample video.
	You will first here the ground truth, then ~3kbps, then 12kbps, for two songs."></a></p>


## What's up?

See [the changelog](CHANGELOG.md) for details on releases.

## Installation

EnCodec requires Python 3.8, and a reasonably recent version of PyTorch (1.11.0 ideally).
To install EnCodec, you can run from this repository:
```bash
pip install -U git+https://git@github.com/facebookresearch/encodec#egg=encodec
# of if you cloned the repo locally
pip install .
```

## Usage

You can then use the EnCodec command, either as
```bash
python3 -m encodec [...]
# or
encodec [...]
```

If you want to directly use the compression API, checkout `encodec.compress`
and `encodec.model`. See hereafter for instructions on how to extract the discrete
representation.

### Model storage

The models will be automatically downloaded on first use using Torch Hub.
For more information on where those models are stored, or how to customize
the storage location, [checkout their documentation.](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved)

### Compression

```bash
encodec [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] INPUT_FILE [OUTPUT_FILE]
```
Given any audio file supported by torchaudio on your platform, compresses
it with EnCodec to the target bandwidth (default is 6 kbps, can be either 1.5, 3, 6, 12 or 24).
OUTPUT_FILE must end in `.ecdc`. If not provided it will be the same as `INPUT_FILE`,
replacing the extension with `.ecdc`.
In order to use the model operating at 48 kHz on stereophonic audio, use the `--hq` flag.
The `-f` flag is used to force overwrite an existing output file.
Use the `--lm` flag to use the pretrained language model with entropy coding (expect it to
be much slower).

If the sample rate or number of channels of the input doesn't match that of the model,
the command will automatically resample / reduce channels as needed.

### Decompression
```bash
encodec [-f] [-r] ENCODEC_FILE [OUTPUT_WAV_FILE]
```
Given a `.ecdc` file previously generated, this will decode it to the given output wav file.
If not provided, the output will default to the input with the `.wav` extension.
Use the `-f` file to force overwrite the output file (be carefull if compress then decompress,
not to overwrite your original file !). Use the `-r` flag if you experience clipping, this will
rescale the output file to avoid it.

### Compression + Decompression
```bash
encodec [-r] [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] INPUT_FILE OUTPUT_WAV_FILE
```
When `OUTPUT_WAV_FILE` has the `.wav` extension (as opposed to `.ecdc`), the `encodec`
command will instead compress and immediately decompress without storing the intermediate
`.ecdc` file.

### Extracting discrete representations

The EnCodec model can also be used to extract discrete representations from the audio waveform.

```python
from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("<PATH_TO_AUDIO_FILE>")
wav = wav.unsqueeze(0)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)

# Extract discrete codes from EnCodec
encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
```

Note that the 48 kHz model processes the audio by chunks of 1 seconds, with an overlap of 1%,
and renormalizes the audio to have unit scale. For this model, the output of `model.encode(wav)`
would a list (for each frame of 1 second) of a tuple `(codes, scale)` with `scale` a scalar tensor.

## Installation for development

This will install the dependencies and a `encodec` in developer mode (changes to the files
will directly reflect), along with the dependencies to run unit tests.
```
pip install -e '.[dev]'
```

### Test

You can run the unit tests with
```
make tests
```

## Citation

If you use this code or results in your paper, please cite our work as:

```
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```

## License

This repository is released under the CC-BY-NC 4.0. license as found in the
[LICENSE](LICENSE) file.
