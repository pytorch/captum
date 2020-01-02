#!/usr/bin/env python3
import random
import unittest

import numpy as np
import torch


def assertArraysAlmostEqual(inputArr, refArr, delta=0.05):
    for index, (input, ref) in enumerate(zip(inputArr, refArr)):
        almost_equal = abs(input - ref) <= delta
        if hasattr(almost_equal, "__iter__"):
            almost_equal = almost_equal.all()
        assert (
            almost_equal
        ), "Values at index {}, {} and {}, \
            differ more than by {}".format(
            index, input, ref, delta
        )


def assertTensorAlmostEqual(test, actual, expected, delta=0.0001, mode="sum"):
    assert isinstance(actual, torch.Tensor), (
        "Actual parameter given for " "comparison must be a tensor."
    )
    actual = actual.squeeze()
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    if mode == "sum":
        test.assertAlmostEqual(
            torch.sum(torch.abs(actual - expected)).item(), 0.0, delta=delta
        )
    elif mode == "max":
        test.assertAlmostEqual(
            torch.max(torch.abs(actual - expected)).item(), 0.0, delta=delta
        )
    else:
        raise ValueError("Mode for assertion comparison must be one of `max` or `sum`.")


def assertTensorTuplesAlmostEqual(test, actual, expected, delta=0.0001, mode="sum"):
    if isinstance(expected, tuple):
        for i in range(len(expected)):
            assertTensorAlmostEqual(test, actual[i], expected[i], delta, mode)
    else:
        assertTensorAlmostEqual(test, actual, expected, delta, mode)


def assertAttributionComparision(test, attributions1, attributions2):
    for attribution1, attribution2 in zip(attributions1, attributions2):
        for attr_row1, attr_row2 in zip(
            attribution1.detach().numpy(), attribution2.detach().numpy()
        ):
            if isinstance(attr_row1, np.ndarray):
                assertArraysAlmostEqual(attr_row1, attr_row2, delta=0.05)
            else:
                test.assertAlmostEqual(attr_row1, attr_row2, delta=0.05)


def assert_delta(test, delta):
    delta_condition = all(abs(delta.numpy().flatten()) < 0.00001)
    test.assertTrue(
        delta_condition,
        "The sum of attribution values {} for relu layer is not "
        "nearly equal to the difference between the endpoint for "
        "some samples".format(delta),
    )


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed. Since many torch
    initializations are random, this ensures that tests run deterministically.
    """

    def setUp(self):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True


class BaseGPUTest(BaseTest):
    """
    This class provides a basic framework for all Captum tests requiring
    CUDA and available GPUs to run appropriately, such as tests for
    DataParallel models. If CUDA is not available, these tests are skipped.
    """

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")
