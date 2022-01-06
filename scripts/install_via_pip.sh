#!/bin/bash

set -e

PYTORCH_NIGHTLY=false
DEPLOY=false
CHOSEN_TORCH_VERSION=-1

while getopts 'ndfv:' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHTLY=true ;;
    d) DEPLOY=true ;;
    f) FRAMEWORKS=true ;;
    v) CHOSEN_TORCH_VERSION=${OPTARG};;
    *) echo "usage: $0 [-n] [-d] [-f] [-v version]" >&2
       exit 1 ;;
    esac
  done

# NOTE: Only Debian variants are supported, since this script is only
# used by our tests on CircleCI. In the future we might generalize,
# but users should hopefully be using conda installs.

# install nodejs and yarn for insights build
sudo apt-get update
sudo apt install apt-transport-https ca-certificates
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install nodejs
sudo apt install yarn

# yarn needs terminal info
export TERM=xterm

# NOTE: All of the below installs use sudo, b/c otherwise pip will get
# permission errors installing in the docker container. An alternative would be
# to use a virtualenv, but that would lead to bifurcation of the CircleCI config
# since we'd need to source the environemnt in each step.

# upgrade pip
sudo pip install --upgrade pip

# install captum with dev deps
sudo pip install -e .[dev]
sudo BUILD_INSIGHTS=1 python setup.py develop

# install other frameworks if asked for and make sure this is before pytorch
if [[ $FRAMEWORKS == true ]]; then
  sudo pip install pytext-nlp
fi

# install pytorch nightly if asked for
if [[ $PYTORCH_NIGHTLY == true ]]; then
  sudo pip install --upgrade --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
else
  # If no version specified, upgrade to latest release.
  if [[ $CHOSEN_TORCH_VERSION == -1 ]]; then
    sudo pip install --upgrade torch
  else
    sudo pip install torch==$CHOSEN_TORCH_VERSION
  fi
fi

# install deployment bits if asked for
if [[ $DEPLOY == true ]]; then
  sudo pip install beautifulsoup4 ipython nbconvert==5.6.1
fi
