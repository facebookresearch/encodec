# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.2a1] - TBD

Adding balancer for reference.

Convert Paths to str before giving to torchaudio due to [some compat issue with Windows](https://github.com/facebookresearch/encodec/issues/13).
Installing is another way to solve the issue.

Fix bug in convert audio that would not work properly with shapes [*, C, T].

Fixing incorrect order of operations when evaluating the number of frames (thanks @chenjiasheng  for the report).

## [0.1.1] - 2022-10-25

Removed useless warning when using `-r` option.

## [0.1.0] - 2022-10-25

Initial release.
