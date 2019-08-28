from __future__ import print_function

import unittest

import torch
from captum.attr._core.layer_gradient_x_activation import LayerGradientXActivation

from .helpers.basic_models import TestModel_MultiLayer
from .helpers.utils import assertArraysAlmostEqual


class Test(unittest.TestCase):
    def test_simple_input_gradient_activation(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._layer_activation_test_helper(net, net.linear0, inp, [0.0, 400.0, 0.0])

    def test_simple_linear_gradient_activation(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._layer_activation_test_helper(
            net, net.linear1, inp, [90.0, 101.0, 101.0, 101.0]
        )

    def test_simple_relu_gradient_activation(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._layer_activation_test_helper(net, net.relu, inp, [0.0, 8.0, 8.0, 8.0])

    def test_simple_output_gradient_activation(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._layer_activation_test_helper(net, net.linear2, inp, [392.0, 0.0])

    def _layer_activation_test_helper(
        self, model, target_layer, test_input, expected_activation
    ):
        layer_act = LayerGradientXActivation(model, target_layer)
        attributions = layer_act.attribute(test_input, target=0)
        assertArraysAlmostEqual(
            attributions.squeeze(0).tolist(), expected_activation, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
