from setuptools import find_packages, setup

setup(
    name="harl",
    version="1.0.0",
    author="XXX",
    description="PyTorch implementation of HADT Algorithms",
    url="",
    packages=find_packages(),
    license="MIT",
    python_requires="=3.10",
    install_requires=[
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
