#!/usr/bin/env python3

import os
import re
import sys

from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


TEST_REQUIRES = [
    "pytest",
    "pytest-cov",
    "ipywidgets",
    "ipython",
    "jupyter",
    "matplotlib",
]

DEV_REQUIRES = TEST_REQUIRES + ["black", "flake8", "sphinx", "sphinx-autodoc-typehints"]

TUTORIALS_REQUIRES = [
    "ipywidgets",
    "ipython",
    "jupyter",
    "matplotlib",
    "pytext-nlp",
    "torchvision",
]

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "captum/__init__.py"), "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="captum",
    version=version,
    description="Model interpretability for PyTorch",
    author="PyTorch Core Team",
    license="BSD",
    url="https://captum.ai",
    project_urls={
        "Documentation": "https://captum.ai",
        "Source": "https://github.com/pytorch/captum",
        "conda": "https://anaconda.org/pytorch/captum",
    },
    keywords=[
        "Model Interpretability",
        "Model Understanding",
        "Feature Importance",
        "Neuron Importance",
        "PyTorch",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=["numpy", "torch>=1.2"],
    packages=find_packages(),
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
