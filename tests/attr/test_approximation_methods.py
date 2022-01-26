#!/usr/bin/env python3

import unittest

import torch
from captum.attr._utils.approximation_methods import Riemann, riemann_builders
from tests.helpers.basic import assertTensorAlmostEqual


class Test(unittest.TestCase):
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)

    def test_riemann_0(self):
        with self.assertRaises(AssertionError):
            step_sizes, alphas = riemann_builders()
            step_sizes(0)
            alphas(0)

    def test_riemann_2(self):
        expected_step_sizes_lrm = [0.5, 0.5]
        expected_step_sizes_trapezoid = [0.25, 0.25]
        expected_left = [0.0, 0.5]
        expected_right = [0.5, 1.0]
        expected_middle = [0.25, 0.75]
        expected_trapezoid = [0.0, 1.0]
        self._assert_steps_and_alphas(
            2,
            expected_step_sizes_lrm,
            expected_step_sizes_trapezoid,
            expected_left,
            expected_right,
            expected_middle,
            expected_trapezoid,
        )

    def test_riemann_3(self):
        expected_step_sizes = [1 / 3] * 3
        expected_step_sizes_trapezoid = [1 / 6, 1 / 3, 1 / 6]
        expected_left = [0.0, 1 / 3, 2 / 3]
        expected_right = [1 / 3, 2 / 3, 1.0]
        expected_middle = [1 / 6, 0.5, 1 - 1 / 6]
        expected_trapezoid = [0.0, 0.5, 1.0]
        self._assert_steps_and_alphas(
            3,
            expected_step_sizes,
            expected_step_sizes_trapezoid,
            expected_left,
            expected_right,
            expected_middle,
            expected_trapezoid,
        )

    def test_riemann_4(self):
        expected_step_sizes = [1 / 4] * 4
        expected_step_sizes_trapezoid = [1 / 8, 1 / 4, 1 / 4, 1 / 8]
        expected_left = [0.0, 0.25, 0.5, 0.75]
        expected_right = [0.25, 0.5, 0.75, 1.0]
        expected_middle = [0.125, 0.375, 0.625, 0.875]
        expected_trapezoid = [0.0, 1 / 3, 2 / 3, 1.0]
        self._assert_steps_and_alphas(
            4,
            expected_step_sizes,
            expected_step_sizes_trapezoid,
            expected_left,
            expected_right,
            expected_middle,
            expected_trapezoid,
        )

    def _assert_steps_and_alphas(
        self,
        n,
        expected_step_sizes,
        expected_step_sizes_trapezoid,
        expected_left,
        expected_right,
        expected_middle,
        expected_trapezoid,
    ):
        step_sizes_left, alphas_left = riemann_builders(Riemann.left)
        step_sizes_right, alphas_right = riemann_builders(Riemann.right)
        step_sizes_middle, alphas_middle = riemann_builders(Riemann.middle)
        step_sizes_trapezoid, alphas_trapezoid = riemann_builders(Riemann.trapezoid)
        assertTensorAlmostEqual(
            self,
            torch.tensor(expected_step_sizes),
            step_sizes_left(n),
            delta=0.05,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            torch.tensor(expected_step_sizes),
            step_sizes_right(n),
            delta=0.05,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            torch.tensor(expected_step_sizes),
            step_sizes_middle(n),
            delta=0.05,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            torch.tensor(expected_step_sizes_trapezoid),
            step_sizes_trapezoid(n),
            delta=0.05,
            mode="max",
        )
        assertTensorAlmostEqual(
            self, torch.tensor(expected_left), alphas_left(n), delta=0.05, mode="max"
        )
        assertTensorAlmostEqual(
            self, torch.tensor(expected_right), alphas_right(n), delta=0.05, mode="max"
        )
        assertTensorAlmostEqual(
            self,
            torch.tensor(expected_middle),
            alphas_middle(n),
            delta=0.05,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            torch.tensor(expected_trapezoid),
            alphas_trapezoid(n),
            delta=0.05,
            mode="max",
        )


# TODO write a test case for gauss-legendre
