from __future__ import print_function

import unittest

import torch
from captum.attr._core.modified_relu_grad import GuidedBackprop
from captum.attr._core.neuron_modified_relu_grad import NeuronGuidedBackprop

from .helpers.basic_models import BasicModel_ConvNet_One_Conv
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_conv_gb(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        exp = [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 3.0, 2.0],
            [1.0, 3.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._guided_backprop_test_assert(net, (inp,), (exp,))

    def test_simple_input_conv_neuron_gb(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        exp = [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 3.0, 2.0],
            [1.0, 3.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._neuron_guided_backprop_test_assert(net, net.fc1, (0,), (inp,), (exp,))

    def test_simple_multi_input_conv_gb(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [1.0, 2.0, 2.0, 1.0],
            [2.0, 4.0, 4.0, 2.0],
            [2.0, 4.0, 4.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._guided_backprop_test_assert(net, (inp, inp2), (ex_attr, ex_attr))

    def test_simple_multi_input_conv_neuron_gb(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [1.0, 2.0, 2.0, 1.0],
            [2.0, 4.0, 4.0, 2.0],
            [2.0, 4.0, 4.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._neuron_guided_backprop_test_assert(
            net, net.fc1, (3,), (inp, inp2), (ex_attr, ex_attr)
        )

    def test_gb_matching(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 100.0 * torch.randn(1, 1, 4, 4)
        self._guided_backprop_matching_assert(net, net.relu2, inp)

    def _guided_backprop_test_assert(
        self, model, test_input, expected, additional_input=None
    ):
        guided_backprop = GuidedBackprop(model)
        attributions = guided_backprop.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)

    def _neuron_guided_backprop_test_assert(
        self, model, layer, neuron_index, test_input, expected, additional_input=None
    ):
        guided_backprop = NeuronGuidedBackprop(model, layer)
        attributions = guided_backprop.attribute(
            test_input,
            neuron_index=neuron_index,
            additional_forward_args=additional_input,
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)

    def _guided_backprop_matching_assert(self, model, output_layer, test_input):
        out = model(test_input)
        attrib = GuidedBackprop(model)
        neuron_attrib = NeuronGuidedBackprop(model, output_layer)
        for i in range(out.shape[1]):
            gbp_vals = attrib.attribute(test_input, target=i)
            neuron_gbp_vals = neuron_attrib.attribute(test_input, (i,))
            assertTensorAlmostEqual(self, gbp_vals, neuron_gbp_vals, delta=0.01)


if __name__ == "__main__":
    unittest.main()
