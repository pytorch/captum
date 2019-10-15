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
conda update --force -y -n base -c defaults conda

# required to use conda develop
conda install -y conda-build

# install yarn for insights build
conda install -y -c conda-forge yarn

# install other frameworks if asked for and make sure this is before pytorch
if [[ $FRAMEWORKS == true ]]; then
  pip install pytext-nlp
fi

if [[ $PYTORCH_NIGHTLY == true ]]; then
  # install CPU version for much smaller download
  conda install -y -c pytorch pytorch-nightly-cpu
else
 # install CPU version for much smaller download
 conda install -y -c pytorch pytorch-cpu
fi

# install other deps
conda install -y numpy sphinx pytest flake8 ipywidgets ipython
conda install -y -c conda-forge black matplotlib pytest-cov sphinx-autodoc-typehints

# build insights and install captum
# TODO: remove CI=false when we want React warnings treated as errors
CI=false BUILD_INSIGHTS=1 python setup.py develop
