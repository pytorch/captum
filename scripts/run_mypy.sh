#!/bin/bash
set -e

# This runs mypy's static type checker on parts of Captum supporting type
# hints.

mypy -p captum.attr --ignore-missing-imports --allow-redefinition
mypy -p captum.insights --ignore-missing-imports --allow-redefinition
mypy -p captum.metrics --ignore-missing-imports --allow-redefinition
mypy -p captum.robust --ignore-missing-imports --allow-redefinition
mypy -p captum.concept --ignore-missing-imports --allow-redefinition
mypy -p tests --ignore-missing-imports --allow-redefinition
mypy -p captum._utils --ignore-missing-imports --allow-redefinition
