# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark script."""


import io
import time
import torch
import torchaudio

from encodec.model import EncodecModel
from encodec.quantization.ac import ArithmeticCoder, ArithmeticDecoder, build_stable_quantized_cdf


def _timer():
    last = time.perf_counter()

    def _measure():
        nonlocal last
        result = time.perf_counter() - last
        last += result
        return result

    return _measure


def main():
    torch.set_num_threads(1)
    model_lq = EncodecModel.encodec_model_24khz()
    model_hq = EncodecModel.encodec_model_48khz()

    for model in [model_lq, model_hq]:
        sr = model.sample_rate // 1000
        x, _ = torchaudio.load(f'test_{sr}k.wav')
        x = x[None, :, :model.sample_rate * 10]
        model.set_target_bandwidth(12)
        lm = model.get_lm_model()

        timer = _timer()
        with torch.no_grad():
            frames = model.encode(x)
        print("Time to encode: ", timer())
        codes = torch.cat([f for f, _ in frames], dim=-1)

        _, K, T = codes.shape
        offset = 0
        input_ = torch.zeros(1, K, 1, dtype=torch.long, device=x.device)
        probas = torch.zeros(1, lm.card, K, T)
        offset = 0
        states = None
        for t in range(T):
            with torch.no_grad():
                probas[:, :, :, t: t + 1], states, offset = lm(input_, states, offset)
            input_ = codes[:, :, t: t + 1] + 1
        print("Time to eval LM:", timer())
        fo = io.BytesIO()
        coder = ArithmeticCoder(fo)
        for t in range(T):
            for k, value in enumerate(codes[0, :, t].tolist()):
                q_cdf = build_stable_quantized_cdf(
                    probas[0, :, k, t], coder.total_range_bits, check=False)
                coder.push(value, q_cdf)
        print("Time to AC enc.:", timer())
        decoder = ArithmeticDecoder(fo)
        for t in range(T):
            for k, value in enumerate(codes[0, :, t].tolist()):
                q_cdf = build_stable_quantized_cdf(
                    probas[0, :, k, t], coder.total_range_bits, check=False)
                decoder.pull(q_cdf)
        print("Time to AC dec.:", timer())
        with torch.no_grad():
            _ = model.decode(frames)
        print("Time to decode:", timer())


if __name__ == '__main__':
    main()
