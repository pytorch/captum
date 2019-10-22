from __future__ import print_function

import unittest

import torch
from captum.attr._core.guided_backprop import Deconvolution
from captum.attr._core.neuron_guided_backprop import NeuronDeconvolution

from .helpers.basic_models import BasicModel_ConvNet_One_Conv
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_conv_deconv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        exp = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._deconv_test_assert(net, (inp,), (exp,))

    def test_simple_input_conv_neuron_deconv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        exp = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._neuron_deconv_test_assert(net, net.fc1, (0,), (inp,), (exp,))

    def test_simple_multi_input_conv_deconv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._deconv_test_assert(net, (inp, inp2), (ex_attr, ex_attr))

    def test_simple_multi_input_conv_neuron_deconv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        self._neuron_deconv_test_assert(
            net, net.fc1, (3,), (inp, inp2), (ex_attr, ex_attr)
        )

    def test_deconv_matching(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 100.0 * torch.randn(1, 1, 4, 4)
        self._deconv_matching_assert(net, net.relu2, inp)

    def _deconv_test_assert(
        self, model, test_input, expected, additional_input=None
    ):
        deconv = Deconvolution(model)
        attributions = deconv.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)

    def _neuron_deconv_test_assert(
        self, model, layer, neuron_index, test_input, expected, additional_input=None
    ):
        deconv = NeuronDeconvolution(model, layer)
        attributions = deconv.attribute(
            test_input,
            neuron_index=neuron_index,
            additional_forward_args=additional_input,
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)

    def _deconv_matching_assert(self, model, output_layer, test_input):
        out = model(test_input)
        attrib = Deconvolution(model)
        neuron_attrib = NeuronDeconvolution(model, output_layer)
        for i in range(out.shape[1]):
            deconv_vals = attrib.attribute(test_input, target=i)
            neuron_deconv_vals = neuron_attrib.attribute(test_input, (i,))
            assertTensorAlmostEqual(self, deconv_vals, neuron_deconv_vals, delta=0.01)


if __name__ == "__main__":
    unittest.main()
