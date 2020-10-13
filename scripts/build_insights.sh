#!/bin/bash

# This builds the Captum Insights React-based UI using yarn.
#
# Run this script from the project root using `./scripts/build_insights.sh`

set -e

usage() {
  echo "Usage: $0"
  echo ""
  echo "Build Captum Insights."
  echo ""
  exit 1
}

while getopts 'h' flag; do
  case "${flag}" in
    h)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

# check if yarn is installed
if [ ! -x "$(command -v yarn)" ]; then
  echo ""
  echo "Please install yarn with 'conda install -c conda-forge yarn' or equivalent."
  echo ""
  exit 1
fi

# go into subdir
pushd captum/insights/attr_vis/frontend || exit

echo "-----------------------------------"
echo "Install Dependencies"
echo "-----------------------------------"
yarn install

echo "-----------------------------------"
echo "Building Captum Insights"
echo "-----------------------------------"
yarn build

pushd widget || exit

echo "-----------------------------------"
echo "Building Captum Insights widget"
echo "-----------------------------------"

../node_modules/.bin/webpack-cli --config webpack.config.js

# exit subdir
popd || exit

# exit subdir
popd || exit
