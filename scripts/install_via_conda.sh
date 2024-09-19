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
conda update -q --all --yes

# required to use conda develop
conda install -q -y conda-build

# install other frameworks if asked for and make sure this is before pytorch
if [[ $FRAMEWORKS == true ]]; then
  pip install -q pytext-nlp
fi

if [[ $PYTORCH_NIGHTLY == true ]]; then
  # install CPU version for much smaller download
  conda install -q -y pytorch cpuonly -c pytorch-nightly
else
 # install CPU version for much smaller download
 conda install -q -y -c pytorch pytorch-cpu
fi

# install other deps
conda install -q -y pytest ipywidgets ipython scikit-learn parameterized werkzeug==2.2.2
conda install -q -y -c conda-forge matplotlib pytest-cov flask flask-compress
conda install -q -y transformers

# install captum
python setup.py develop
