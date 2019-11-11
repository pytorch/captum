#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.internal_influence import InternalInfluence

from .helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertArraysAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_internal_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_assert(net, net.linear0, inp, [[3.9, 3.9, 3.9]])

    def test_simple_linear_internal_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.9, 1.0, 1.0, 1.0]]
        )

    def test_simple_relu_input_internal_inf_inplace(self):
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.relu, inp, [[0.9, 1.0, 1.0, 1.0]], attribute_to_layer_input=True
        )

    def test_simple_linear_internal_inf_inplace(self):
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.9, 1.0, 1.0, 1.0]]
        )

    def test_simple_relu_internal_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._internal_influence_test_assert(net, net.relu, inp, [[1.0, 1.0, 1.0, 1.0]])

    def test_simple_output_internal_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(net, net.linear2, inp, [[1.0, 0.0]])

    def test_simple_with_baseline_internal_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 80.0, 0.0]])
        base = torch.tensor([[0.0, -20.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.7, 0.8, 0.8, 0.8]], base
        )

    def test_simple_multi_input_linear2_internal_inf(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._internal_influence_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            [[1.0, 0.0]],
            additional_args=(4,),
        )

    def test_simple_multi_input_relu_internal_inf(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._internal_influence_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            [[1.0, 1.0, 1.0, 1.0]],
            additional_args=(inp3, 5),
        )

    def test_simple_multi_input_batch_relu_internal_inf(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 80.0, 0.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 20.0, 0.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        self._internal_influence_test_assert(
            net,
            net.model.linear1,
            (inp1, inp2),
            [[0.95, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            additional_args=(inp3, 5),
        )

    def test_multiple_linear_internal_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ],
            requires_grad=True,
        )
        self._internal_influence_test_assert(
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
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 80.0, 0.0], [30.0, 30.0, 0.0]], requires_grad=True)
        base = torch.tensor(
            [[0.0, -20.0, 0.0], [-20.0, -20.0, 0.0]], requires_grad=True
        )
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.7, 0.8, 0.8, 0.8], [0.5, 0.6, 0.6, 0.6]], base
        )

    def _internal_influence_test_assert(
        self,
        model,
        target_layer,
        test_input,
        expected_activation,
        baseline=None,
        additional_args=None,
        attribute_to_layer_input=False,
    ):
        for internal_batch_size in [None, 1, 20]:
            int_inf = InternalInfluence(model, target_layer)
            attributions = int_inf.attribute(
                test_input,
                baselines=baseline,
                target=0,
                n_steps=500,
                method="riemann_trapezoid",
                additional_forward_args=additional_args,
                internal_batch_size=internal_batch_size,
                attribute_to_layer_input=attribute_to_layer_input,
            )
            for i in range(len(expected_activation)):
                assertArraysAlmostEqual(
                    attributions[i : i + 1].squeeze(0).tolist(),
                    expected_activation[i],
                    delta=0.01,
                )


if __name__ == "__main__":
    unittest.main()
