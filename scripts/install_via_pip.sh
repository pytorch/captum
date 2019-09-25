#!/bin/bash

set -ex

PYTORCH_NIGHLTY=false
DEPLOY=false

while getopts 'ndf' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHLTY=true ;;
    d) DEPLOY=true ;;
    f) FRAMEWORKS=true ;;
    *) echo "usage: $0 [-n] [-d] [-f]" >&2
       exit 1 ;;
    esac
  done

# install yarn for insights build
# and make sure cmdtest doesn't get installed instead
sudo apt remove cmdtest
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install yarn

# yarn needs terminal info
export TERM=xterm

# NOTE: All of the below installs use sudo, b/c otherwise pip will get
# permission errors installing in the docker container. An alternative would be
# to use a virtualenv, but that would lead to bifurcation of the CircleCI config
# since we'd need to source the environemnt in each step.

# upgrade pip
sudo pip install --upgrade pip

# install captum + dev deps
sudo pip install -e .[dev]

# install other frameworks if asked for and make sure this is before pytorch
if [[ $FRAMEWORKS == true ]]; then
  sudo pip install pytext-nlp
fi

# install pytorch nightly if asked for
if [[ $PYTORCH_NIGHLTY == true ]]; then
  sudo pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
fi

# install deployment bits if asked for
if [[ $DEPLOY == true ]]; then
  sudo pip install beautifulsoup4 ipython nbconvert
fi
