from __future__ import print_function

import unittest

import torch
from captum.attributions.layer_conductance import LayerConductance
from captum.attributions.neuron_conductance import NeuronConductance

from .helpers.basic_models import TestModel_ConvNet, TestModel_MultiLayer
from .helpers.utils import assertArraysAlmostEqual


class Test(unittest.TestCase):
    def test_simple_conductance_input_linear2(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_input_test_assert(
            net, net.linear2, inp, (0,), [0.0, 390.0, 0.0]
        )

    def test_simple_conductance_input_linear1(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_input_test_assert(net, net.linear1, inp, 0, [0.0, 90.0, 0.0])

    def test_simple_conductance_input_relu(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 70.0, 30.0]], requires_grad=True)
        self._conductance_input_test_assert(net, net.relu, inp, (3,), [0.0, 70.0, 30.0])

    def test_matching_conv2_multi_input_conductance(self):
        net = TestModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.conv2, inp)

    def test_matching_relu2_multi_input_conductance(self):
        net = TestModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.relu2, inp, baseline)

    def test_matching_pool2_multi_input_conductance(self):
        net = TestModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.pool2, inp, baseline)

    def _conductance_input_test_assert(
        self, model, target_layer, test_input, test_neuron, expected_input_conductance
    ):
        cond = NeuronConductance(model, target_layer)
        attributions = cond.attribute(
            test_input, test_neuron, target=0, n_steps=500, method="gausslegendre"
        )
        assertArraysAlmostEqual(
            attributions.squeeze(0).tolist(), expected_input_conductance, delta=0.1
        )

    def _conductance_input_sum_test_assert(
        self, model, target_layer, test_input, test_baseline=None
    ):
        layer_cond = LayerConductance(model, target_layer)
        attributions = layer_cond.attribute(
            test_input,
            baselines=test_baseline,
            target=0,
            n_steps=500,
            method="gausslegendre",
        )
        neuron_cond = NeuronConductance(model, target_layer)
        for i in range(attributions.shape[1]):
            for j in range(attributions.shape[2]):
                for k in range(attributions.shape[3]):
                    neuron_vals = neuron_cond.attribute(
                        test_input,
                        (i, j, k),
                        baselines=test_baseline,
                        target=0,
                        n_steps=500,
                    )
                    for n in range(attributions.shape[0]):
                        self.assertAlmostEqual(
                            torch.sum(neuron_vals[n]),
                            attributions[n, i, j, k],
                            delta=0.005,
                        )


if __name__ == "__main__":
    unittest.main()
