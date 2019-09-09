from __future__ import print_function

import unittest

import torch
from captum.attr._core.internal_influence import InternalInfluence
from captum.attr._core.layer_activation import LayerActivation
from captum.attr._core.layer_conductance import LayerConductance
from captum.attr._core.layer_gradient_x_activation import LayerGradientXActivation

from captum.attr._core.neuron_conductance import NeuronConductance
from captum.attr._core.neuron_gradient import NeuronGradient
from captum.attr._core.neuron_integrated_gradients import NeuronIntegratedGradients

from .helpers.basic_models import TestModel_MultiLayer, TestModel_MultiLayer_MultiInput
from .helpers.utils import assertArraysAlmostEqual, BaseGPUTest


class Test(BaseGPUTest):
    def test_simple_input_internal_inf(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(InternalInfluence, net, net.relu, inputs=inp, target=0)

    def test_multi_input_internal_inf(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(InternalInfluence, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5), target=0)

    def test_simple_layer_activation(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(LayerActivation, net, net.relu, inputs=inp)

    def test_multi_input_layer_activation(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(LayerActivation, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5))

    def test_simple_layer_conductance(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(LayerConductance, net, net.relu, inputs=inp, target=1)

    def test_multi_input_layer_conductance(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(LayerConductance, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5), target=1)

    def test_simple_layer_gradient_x_activation(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(LayerGradientXActivation, net, net.relu, inputs=inp, target=1)

    def test_multi_input_layer_gradient_x_activation(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(LayerGradientXActivation, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5), target=1)

    def test_simple_neuron_conductance(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(NeuronConductance, net, net.relu, inputs=inp, neuron_index=(3,), target=1)

    def test_multi_input_neuron_conductance(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(NeuronConductance, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5), target=1, neuron_index=(3,))

    def test_simple_neuron_gradient(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(NeuronGradient, net, net.relu, inputs=inp, neuron_index=(3,))

    def test_multi_input_neuron_gradient(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(NeuronGradient, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5), neuron_index=(3,))

    def test_simple_neuron_integrated_gradient(self):
        net = TestModel_MultiLayer().cuda()
        inp = torch.tensor([[0.0, 100.0, 0.0],[20.0, 100.0, 120.0],[30.0, 10.0, 0.0],[0.0, 0.0, 2.0]]).cuda()
        self._data_parallel_test_assert(NeuronIntegratedGradients, net, net.relu, inputs=inp, neuron_index=(3,))

    def test_multi_input_integrated_gradient(self):
        net = TestModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = 10*torch.randn(12,3).cuda(), 5*torch.randn(12,3).cuda(), 2*torch.randn(12,3).cuda()
        self._data_parallel_test_assert(NeuronIntegratedGradients, net, net.model.relu, inputs=(inp1, inp2), additional_forward_args=(inp3, 5), neuron_index=(3,))

    def _data_parallel_test_assert(
        self,
        method,
        model,
        target_layer=None,
        contains_delta=False,
        **kwargs,
    ):
        dp_model = torch.nn.parallel.DataParallel(model)
        if target_layer:
            attr_orig = method(model, target_layer)
            attr_dp = method(dp_model, target_layer)
        else:
            attr_orig = method(model)
            attr_dp = method(dp_model)
        attributions_orig = attr_orig.attribute(**kwargs)
        attributions_dp = attr_dp.attribute(**kwargs)
        if contains_delta:
            attributions_orig = attributions_orig[0]
            attributions_dp = attributions_dp[0]
        if isinstance(attributions_dp, torch.Tensor):
            self.assertAlmostEqual(torch.sum(torch.abs(attributions_orig - attributions_dp)),0,delta=0.0001)
        else:
            for i in range(len(attributions_orig)):
                self.assertAlmostEqual(torch.sum(torch.abs(attributions_orig[i] - attributions_dp[i])),0,delta=0.0001)


if __name__ == "__main__":
    unittest.main()
