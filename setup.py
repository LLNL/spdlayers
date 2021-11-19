# Copyright 2021, Lawrence Livermore National Security, LLC and spdlayer
# contributors
# SPDX-License-Identifier: MIT

from distutils.core import setup
from pathlib import Path
import spdlayers.version as spdlayers
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
short = "Symmetric Positive Definite (SPD) enforcement layers for PyTorch"

setup(
    name="spdlayers",
    version=spdlayers.__version__,
    author="Charles Jekel",
    author_email="jekel1@llnl.gov",
    packages=['spdlayers'],
    description=short,
    license='MIT License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms=["any"],
    install_requires=[
        "torch >= 1.9.0",
        "setuptools >= 38.6.0"
        ],
    python_requires=">=3.6",  # needed for @ as matrix multiplication
    classifiers=[
          'License :: OSI Approved :: Python Software Foundation License',
          'Programming Language :: Python :: 3',
        ],
)
