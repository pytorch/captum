#!/usr/bin/env python3
from __future__ import print_function

import unittest

import torch
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.neuron_integrated_gradients import NeuronIntegratedGradients

from .helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertArraysAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_ig_input_linear2(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._ig_input_test_assert(net, net.linear2, inp, 0, [0.0, 390.0, 0.0])

    def test_simple_ig_input_linear1(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.linear1, inp, (0,), [0.0, 100.0, 0.0])

    def test_simple_ig_input_relu(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 6.0, 14.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.relu, inp, (0,), [0.0, 3.0, 7.0])

    def test_simple_ig_input_relu2(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._ig_input_test_assert(net, net.relu, inp, 1, [0.0, 5.0, 4.0])

    def test_simple_ig_multi_input_linear2(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            (0,),
            ([[0.0, 156.0, 0.0]], [[0.0, 156.0, 0.0]], [[0.0, 78.0, 0.0]]),
            (4,),
        )

    def test_simple_ig_multi_input_relu(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (0,),
            ([[0.0, 1.5, 3.5]], [[0.0, 1.5, 3.5]]),
            (inp3, 0.5),
        )

    def test_simple_ig_multi_input_relu_batch(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 80.0, 0.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 20.0, 0.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (0,),
            ([[0.0, 1.5, 3.5], [0.0, 40.0, 0.0]], [[0.0, 1.5, 3.5], [0.0, 10.0, 0.0]]),
            (inp3, 0.5),
        )

    def test_matching_output_gradient(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10, 10, requires_grad=True)
        self._ig_matching_test_assert(net, net.softmax, inp, baseline)

    def _ig_input_test_assert(
        self,
        model,
        target_layer,
        test_input,
        test_neuron,
        expected_input_ig,
        additional_input=None,
    ):
        for internal_batch_size in [None, 1, 20]:
            grad = NeuronIntegratedGradients(model, target_layer)
            attributions = grad.attribute(
                test_input,
                test_neuron,
                n_steps=500,
                method="gausslegendre",
                additional_forward_args=additional_input,
                internal_batch_size=internal_batch_size,
            )
            if isinstance(expected_input_ig, tuple):
                for i in range(len(expected_input_ig)):
                    for j in range(attributions[i].shape[0]):
                        assertArraysAlmostEqual(
                            attributions[i][j].squeeze(0).tolist(),
                            expected_input_ig[i][j],
                            delta=0.1,
                        )
            else:
                assertArraysAlmostEqual(
                    attributions.squeeze(0).tolist(), expected_input_ig, delta=0.1
                )

    def _ig_matching_test_assert(self, model, output_layer, test_input, baseline=None):
        out = model(test_input)
        input_attrib = IntegratedGradients(model)
        ig_attrib = NeuronIntegratedGradients(model, output_layer)
        for i in range(out.shape[1]):
            ig_vals = input_attrib.attribute(test_input, target=i, baselines=baseline)
            neuron_ig_vals = ig_attrib.attribute(test_input, (i,), baselines=baseline)
            assertArraysAlmostEqual(
                ig_vals.reshape(-1).tolist(),
                neuron_ig_vals.reshape(-1).tolist(),
                delta=0.001,
            )
            self.assertEqual(neuron_ig_vals.shape, test_input.shape)


if __name__ == "__main__":
    unittest.main()
