#!/usr/bin/env python3
from __future__ import print_function

import unittest

import torch
from captum.attr._core.layer_gradient_x_activation import LayerGradientXActivation

from .helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertArraysAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_gradient_activation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._layer_activation_test_assert(net, net.linear0, inp, [0.0, 400.0, 0.0])

    def test_simple_linear_gradient_activation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.linear1, inp, [90.0, 101.0, 101.0, 101.0]
        )

    def test_simple_relu_gradient_activation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._layer_activation_test_assert(net, net.relu, inp, [0.0, 8.0, 8.0, 8.0])

    def test_simple_output_gradient_activation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._layer_activation_test_assert(net, net.linear2, inp, [392.0, 0.0])

    def test_simple_gradient_activation_multi_input_linear2(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.model.linear2, (inp1, inp2, inp3), [392.0, 0.0], (4,)
        )

    def test_simple_gradient_activation_multi_input_relu(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.model.relu, (inp1, inp2), [90.0, 101.0, 101.0, 101.0], (inp3, 5)
        )

    def _layer_activation_test_assert(
        self,
        model,
        target_layer,
        test_input,
        expected_activation,
        additional_input=None,
    ):
        layer_act = LayerGradientXActivation(model, target_layer)
        attributions = layer_act.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        assertArraysAlmostEqual(
            attributions.squeeze(0).tolist(), expected_activation, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
