#!/usr/bin/env python3
import copy
import random
import unittest
from typing import Callable

import numpy as np
import torch

from captum.log import patch_methods


def deep_copy_args(func: Callable):
    def copy_args(*args, **kwargs):
        return func(
            *(copy.deepcopy(x) for x in args),
            **{k: copy.deepcopy(v) for k, v in kwargs.items()}
        )

    return copy_args


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
    actual = actual.squeeze().cpu()
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    expected = expected.squeeze().cpu()
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


def set_all_random_seeds(seed):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed. Since many torch
    initializations are random, this ensures that tests run deterministically.
    """

    def setUp(self):
        set_all_random_seeds(1234)
        patch_methods(self)
