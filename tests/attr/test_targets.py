from __future__ import print_function

import unittest

import torch
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.saliency import Saliency
from captum.attr._core.input_x_gradient import InputXGradient

from captum.attr._core.internal_influence import InternalInfluence
from captum.attr._core.layer_activation import LayerActivation
from captum.attr._core.layer_conductance import LayerConductance
from captum.attr._core.layer_gradient_x_activation import LayerGradientXActivation

from captum.attr._core.neuron_conductance import NeuronConductance
from captum.attr._core.neuron_gradient import NeuronGradient
from captum.attr._core.neuron_integrated_gradients import NeuronIntegratedGradients

from .helpers.basic_models import (
    TestModel_MultiLayer,
    TestModel_MultiLayer_MultiInput,
    TestModel_ConvNet,
)
from .helpers.utils import BaseTest, assertTensorAlmostEqual



class Test(BaseTest):
    def test_simple_target_error(self):
        net = TestModel_MultiLayer()
        inp = torch.zeros((1,3))
        with self.assertRaises(AssertionError):
            attr = IntegratedGradients(net)
            attr.attribute(inp,)

    def test_multi_target_error(self):
        net = TestModel_MultiLayer()
        inp = torch.zeros((1,3))
        with self.assertRaises(AssertionError):
            attr = IntegratedGradients(net)
            attr.attribute(inp,additional_forward_args=(None,True),target=(1,0))

    def test_simple_target_ig(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            IntegratedGradients, net, inputs=inp, targets=[0, 1, 1, 0], test_batches=True,
        )

    def test_multi_target_ig(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            IntegratedGradients, net, inputs=inp, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)], test_batches=True,
        )

    def test_simple_target_saliency(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            Saliency, net, inputs=inp, targets=[0, 1, 1, 0],
            )

    def test_multi_target_saliency(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            Saliency, net, inputs=inp, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)],
        )

    def test_simple_target_input_x_gradient(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            InputXGradient, net, inputs=inp, targets=[0, 1, 1, 0],
            )

    def test_multi_target_input_x_gradient(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            InputXGradient, net, inputs=inp, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)],
        )

    def test_simple_target_int_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            InternalInfluence, net, inputs=inp, target_layer=net.relu, targets=[0, 1, 1, 0], test_batches=True,
        )

    def test_multi_target_int_inf(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            InternalInfluence, net, inputs=inp, target_layer=net.relu, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)], test_batches=True,
        )

    def test_simple_target_layer_cond(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            LayerConductance, net, inputs=inp, target_layer=net.relu, targets=[0, 1, 1, 0], test_batches=True,
        )

    def test_multi_target_layer_cond(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            LayerConductance, net, inputs=inp, target_layer=net.relu, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)], test_batches=True,
        )

    def test_simple_target_layer_gradientx_act(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            LayerGradientXActivation, net, inputs=inp, target_layer=net.relu, targets=[0, 1, 1, 0],
        )

    def test_multi_target_layer_gradientx_act(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            LayerGradientXActivation, net, inputs=inp, target_layer=net.relu, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)],
        )

    def test_simple_target_neuron_conductance(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            NeuronConductance, net, inputs=inp, target_layer=net.relu, targets=[0, 1, 1, 0], neuron_index=3, test_batches=True,
        )

    def test_multi_target_neuron_conductance(self):
        net = TestModel_MultiLayer()
        inp = torch.randn(4,3)
        self._target_batch_test_assert(
            NeuronConductance, net, inputs=inp, target_layer=net.relu, additional_forward_args=(None, True), targets=[(1,0,0), (0,1,1), (1,1,1), (0,0,0)], neuron_index=3, test_batches=True,
        )

    def _target_batch_test_assert(
        self,
        algorithm,
        model,
        inputs,
        targets,
        target_layer=None,
        test_batches=False,
        **kwargs
    ):
        if target_layer:
            attr = algorithm(model, target_layer)
        else:
            attr = algorithm(model)

        batch_sizes = [None]
        if test_batches:
            batch_sizes = [None, 2, 4]
        for batch_size in batch_sizes:
            if batch_size:
                attributions_orig = attr.attribute(inputs=inputs, target=targets,
                    internal_batch_size=batch_size, **kwargs
                )
            else:
                attributions_orig = attr.attribute(inputs=inputs, target=targets, **kwargs)
            if attr._has_convergence_delta():
                attributions_orig = attributions_orig[0]
            for i in range(len(inputs)):
                single_attr = attr.attribute(inputs=inputs[i:i+1], target=targets[i], **kwargs)
                single_attr_target_list = attr.attribute(inputs=inputs[i:i+1], target=targets[i:i+1], **kwargs)
                if attr._has_convergence_delta():
                    single_attr = single_attr[0]
                    single_attr_target_list = single_attr_target_list[0]
                assertTensorAlmostEqual(self, attributions_orig[i:i+1], single_attr)
                assertTensorAlmostEqual(self, attributions_orig[i:i+1], single_attr_target_list)


if __name__ == "__main__":
    unittest.main()
