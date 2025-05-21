#!/bin/bash

set -e

while getopts 'nf' flag; do
  case "${flag}" in
    f) FRAMEWORKS=true ;;
    *) echo "usage: $0 [-n] [-f]" >&2
       exit 1 ;;
    esac
  done

# update conda
# removing due to setuptools error during update
#conda update -y -n base -c defaults conda
conda update -q --all --yes


# install CPU version for much smaller download
conda install -q -y pytorch cpuonly -c pytorch

# install other deps
conda install -q -y pytest ipywidgets ipython scikit-learn parameterized werkzeug
conda install -q -y -c conda-forge matplotlib pytest-cov flask flask-compress conda-build openai
conda install -q -y transformers

# install captum
python setup.py develop
