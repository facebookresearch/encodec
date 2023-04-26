#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Inspired from https://github.com/kennethreitz/setup.py
from pathlib import Path

from setuptools import setup


NAME = 'encodec'
DESCRIPTION = 'High fidelity neural audio codec'
URL = 'https://github.com/facebookresearch/encodec'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez, Jade Copet, Yossi Adi, Gabriel Synnaeve'
REQUIRES_PYTHON = '>=3.8.0'

for line in open('encodec/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['encodec', 'encodec.quantization', 'encodec.modules'],
    extras_require={
        'dev': ['flake8', 'mypy', 'pdoc3'],
    },
    install_requires=['numpy', 'torch', 'torchaudio', 'einops'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['encodec=encodec.__main__:main'],
    },
    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
    ])
