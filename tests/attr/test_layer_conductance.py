#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.layer_conductance import LayerConductance

from .helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.conductance_reference import ConductanceReference
from .helpers.utils import assertArraysAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_conductance(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_test_assert(net, net.linear0, inp, [[0.0, 390.0, 0.0]])

    def test_simple_linear_conductance(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_test_assert(
            net, net.linear1, inp, [[90.0, 100.0, 100.0, 100.0]]
        )

    def test_simple_relu_conductance(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_test_assert(net, net.relu, inp, [[90.0, 100.0, 100.0, 100.0]])

    def test_simple_output_conductance(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_test_assert(net, net.linear2, inp, [[390.0, 0.0]])

    def test_simple_multi_input_linear2_conductance(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._conductance_test_assert(
            net, net.model.linear2, (inp1, inp2, inp3), [[390.0, 0.0]], (4,)
        )

    def test_simple_multi_input_relu_conductance(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._conductance_test_assert(
            net, net.model.relu, (inp1, inp2), [[90.0, 100.0, 100.0, 100.0]], (inp3, 5)
        )

    def test_simple_multi_input_relu_conductance_batch(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0], [0.0, 0.0, 10.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0], [0.0, 0.0, 10.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        self._conductance_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            [[90.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0]],
            (inp3, 5),
        )

    def test_matching_conv1_conductance(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.conv1, inp)

    def test_matching_pool1_conductance(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10)
        self._conductance_reference_test_assert(net, net.pool1, inp)

    def test_matching_conv2_conductance(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.conv2, inp)

    def test_matching_pool2_conductance(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10)
        self._conductance_reference_test_assert(net, net.pool2, inp)

    def test_matching_conv_multi_input_conductance(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(4, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.relu3, inp)

    def test_matching_conv_with_baseline_conductance(self):
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10)
        baseline = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.fc1, inp, baseline)

    def _conductance_test_assert(
        self,
        model,
        target_layer,
        test_input,
        expected_conductance,
        additional_args=None,
    ):
        cond = LayerConductance(model, target_layer)
        for internal_batch_size in (None, 1, 20):
            attributions, delta = cond.attribute(
                test_input,
                target=0,
                n_steps=500,
                method="gausslegendre",
                additional_forward_args=additional_args,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )
            delta_condition = all(abs(delta.numpy().flatten()) < 0.01)
            self.assertTrue(
                delta_condition,
                "Sum of attributions does {}"
                " not match the difference of endpoints.".format(delta),
            )

            for i in range(len(expected_conductance)):
                assertArraysAlmostEqual(
                    attributions[i : i + 1].squeeze(0).tolist(),
                    expected_conductance[i],
                    delta=0.1,
                )

    def _conductance_reference_test_assert(
        self, model, target_layer, test_input, test_baseline=None
    ):
        layer_output = None

        def forward_hook(module, inp, out):
            nonlocal layer_output
            layer_output = out

        hook = target_layer.register_forward_hook(forward_hook)
        final_output = model(test_input)
        hook.remove()
        target_index = torch.argmax(torch.sum(final_output, 0))
        cond = LayerConductance(model, target_layer)
        cond_ref = ConductanceReference(model, target_layer)
        attributions, delta = cond.attribute(
            test_input,
            baselines=test_baseline,
            target=target_index,
            n_steps=300,
            method="gausslegendre",
            return_convergence_delta=True,
        )
        delta_condition = all(abs(delta.numpy().flatten()) < 0.005)
        self.assertTrue(
            delta_condition,
            "Sum of attribution values does {} "
            " not match the difference of endpoints.".format(delta),
        )

        attributions_reference = cond_ref.attribute(
            test_input,
            baselines=test_baseline,
            target=target_index,
            n_steps=300,
            method="gausslegendre",
        )

        # Check that layer output size matches conductance size.
        self.assertEqual(layer_output.shape, attributions.shape)
        # Check that reference implementation output matches standard implementation.
        assertArraysAlmostEqual(
            attributions.reshape(-1).tolist(),
            attributions_reference.reshape(-1).tolist(),
            delta=0.07,
        )

        # Test if batching is working correctly for inputs with multiple examples
        if test_input.shape[0] > 1:
            for i in range(test_input.shape[0]):
                single_attributions = cond.attribute(
                    test_input[i : i + 1],
                    baselines=test_baseline[i : i + 1]
                    if test_baseline is not None
                    else None,
                    target=target_index,
                    n_steps=300,
                    method="gausslegendre",
                )
                # Verify that attributions when passing example independently
                # matches corresponding attribution of batched input.
                assertArraysAlmostEqual(
                    attributions[i : i + 1].reshape(-1).tolist(),
                    single_attributions.reshape(-1).tolist(),
                    delta=0.01,
                )


if __name__ == "__main__":
    unittest.main()
