from __future__ import print_function

import unittest

import torch
from captum.attr._core.internal_influence import InternalInfluence

from .helpers.basic_models import TestModel_MultiLayer
from .helpers.utils import assertArraysAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_helper(net, net.linear0, inp, [[3.9, 3.9, 3.9]])

    def test_simple_linear_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_helper(
            net, net.linear1, inp, [[0.9, 1.0, 1.0, 1.0]]
        )

    def test_simple_relu_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._internal_influence_test_helper(net, net.relu, inp, [[1.0, 1.0, 1.0, 1.0]])

    def test_simple_output_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_helper(net, net.linear2, inp, [[1.0, 0.0]])

    def test_simple_with_baseline_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 80.0, 0.0]], requires_grad=True)
        base = torch.tensor([[0.0, -20.0, 0.0]], requires_grad=True)
        self._internal_influence_test_helper(
            net, net.linear1, inp, [[0.7, 0.8, 0.8, 0.8]], base
        )

    def test_multiple_linear_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ],
            requires_grad=True,
        )
        self._internal_influence_test_helper(
            net,
            net.linear1,
            inp,
            [
                [0.9, 1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0, 1.0],
            ],
        )

    def test_multiple_with_baseline_internal_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 80.0, 0.0], [30.0, 30.0, 0.0]], requires_grad=True)
        base = torch.tensor(
            [[0.0, -20.0, 0.0], [-20.0, -20.0, 0.0]], requires_grad=True
        )
        self._internal_influence_test_helper(
            net, net.linear1, inp, [[0.7, 0.8, 0.8, 0.8], [0.5, 0.6, 0.6, 0.6]], base
        )

    def _internal_influence_test_helper(
        self, model, target_layer, test_input, expected_activation, baseline=None
    ):
        int_inf = InternalInfluence(model, target_layer)
        attributions = int_inf.attribute(
            test_input,
            baselines=baseline,
            target=0,
            n_steps=500,
            method="riemann_trapezoid",
        )
        for i in range(test_input.shape[0]):
            assertArraysAlmostEqual(
                attributions[i : i + 1].squeeze(0).tolist(),
                expected_activation[i],
                delta=0.01,
            )


if __name__ == "__main__":
    unittest.main()
