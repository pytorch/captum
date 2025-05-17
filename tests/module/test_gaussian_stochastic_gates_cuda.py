#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from .test_gaussian_stochastic_gates import TestGaussianStochasticGates


class TestGaussianStochasticGatesCUDA(TestGaussianStochasticGates):
    testing_device: str = "cuda"
