#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from .test_binary_concrete_stochastic_gates import TestBinaryConcreteStochasticGates


class TestBinaryConcreteStochasticGatesCUDA(TestBinaryConcreteStochasticGates):
    testing_device: str = "cuda"
