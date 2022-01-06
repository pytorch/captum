#!/bin/bash

set -e

PYTORCH_NIGHTLY=false

while getopts 'nf' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHTLY=true ;;
    f) FRAMEWORKS=true ;;
    *) echo "usage: $0 [-n] [-f]" >&2
       exit 1 ;;
    esac
  done

# update conda
# removing due to setuptools error during update
#conda update -y -n base -c defaults conda

# required to use conda develop
conda install -y conda-build

# install other frameworks if asked for and make sure this is before pytorch
if [[ $FRAMEWORKS == true ]]; then
  pip install pytext-nlp
fi

if [[ $PYTORCH_NIGHTLY == true ]]; then
  # install CPU version for much smaller download
  conda install -y pytorch cpuonly -c pytorch-nightly
else
 # install CPU version for much smaller download
 conda install -y -c pytorch pytorch-cpu
fi

# install other deps
conda install -y numpy sphinx pytest flake8 ipywidgets ipython scikit-learn
conda install -y -c conda-forge matplotlib pytest-cov sphinx-autodoc-typehints mypy flask flask-compress
# deps not available in conda
pip install sphinxcontrib-katex

# install node/yarn for insights build
conda install -y -c conda-forge yarn
# nodejs should be last, otherwise other conda packages will downgrade node
conda install -y --no-channel-priority -c conda-forge nodejs=14

# install lint deps
pip install black==21.4b2 usort==0.6.4 ufmt

# build insights and install captum
BUILD_INSIGHTS=1 python setup.py develop
