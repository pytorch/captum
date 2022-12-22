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
conda update --all --yes

# required to use conda develop
conda install -y conda-build

# Use faster conda solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

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
conda install -y pytest ipywidgets ipython scikit-learn parameterized
conda install -y -c conda-forge matplotlib pytest-cov

# install captum
python setup.py develop
