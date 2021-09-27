from distutils.core import setup

setup(
    name="spdlayers",
    version="0.0.1",
    author="Charles Jekel",
    author_email="jekel1@llnl.gov",
    packages=['spdlayers'],
    description="SPD Layers :)",
    platforms=["any"],
    install_requires=[
        "torch >= 1.9.0",
        ],
    python_requires=">=3.6",  # needed for @ as matrix multiplication
)
