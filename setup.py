# Copyright 2021, Lawrence Livermore National Security, LLC and spdlayer
# contributors
# SPDX-License-Identifier: MIT

from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
short = "Symmetric Positive Definite (SPD) enforcement layers for PyTorch"

version = {}
with open("spdlayers/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="spdlayers",
    version=version["__version__"],
    author="Charles Jekel",
    author_email="jekel1@llnl.gov",
    url='https://github.com/LLNL/spdlayers',
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
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        ],
)
