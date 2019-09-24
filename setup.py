#!/usr/bin/env python3

import os
import re
import subprocess
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


RUN_BUILD_INSIGHTS = True
VERBOSE_SCRIPT = True
for i, arg in enumerate(sys.argv):
    if arg == "clean":
        RUN_BUILD_INSIGHTS = False
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False


def report(*args):
    if VERBOSE_SCRIPT:
        print(*args)
    else:
        pass


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

INSIGHTS_FILE_SUBDIRS = ["insights/frontend/build"]

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "captum/__init__.py"), "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)
    report("-- Building version " + version)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()


# build Captum Insights via yarn if content doesn't exist
def build_insights():
    report("-- Building Captum Insights")
    is_built = all(
        os.path.exists(os.path.join("captum", d)) for d in INSIGHTS_FILE_SUBDIRS
    )
    if is_built:
        report("Skipping since " + str(INSIGHTS_FILE_SUBDIRS) + " all exist")
    else:
        command = "./scripts/build_insights.sh"
        report("Running: " + command)
        subprocess.check_call(command)


# explore paths under root and subdirs to gather package files
def get_package_files(root, subdirs):
    paths = []
    for subroot in subdirs:
        paths.append(os.path.join(subroot, "*"))
        for path, dirs, _ in os.walk(os.path.join(root, subroot)):
            for d in dirs:
                paths.append(os.path.join(path, d, "*")[len(root) + 1 :])
    return paths


if __name__ == "__main__":

    if RUN_BUILD_INSIGHTS:
        build_insights()

    package_files = get_package_files("captum", INSIGHTS_FILE_SUBDIRS)

    setup(
        name="captum",
        version=version,
        description="Model interpretability for PyTorch",
        author="PyTorch Team",
        license="BSD-3",
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
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
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
        package_data={"captum": package_files},
    )
