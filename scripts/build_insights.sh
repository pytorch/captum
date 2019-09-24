#!/bin/bash

# this builds the Captum Insights React-based UI
#
# run this script from the project root using `./scripts/build_insights.sh`

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

# go into subdir
pushd captum/insights/frontend || exit

echo "-----------------------------------"
echo "Install Dependencies"
echo "-----------------------------------"
yarn install

echo "-----------------------------------"
echo "Building Captum Insights"
echo "-----------------------------------"
yarn build

# exit subdir
popd || exit
