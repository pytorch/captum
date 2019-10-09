from __future__ import print_function

import unittest

import torch
from captum.attr._core.grad_cam import LayerGradCam

from .helpers.basic_models import BasicModel_MultiLayer, BasicModel_SmallConvNet
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_non_conv(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._grad_cam_test_assert(net, net.linear0, inp, [400.0])

    def test_simple_input_conv(self):
        net = BasicModel_SmallConvNet()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        self._grad_cam_test_assert(net, net.conv1, inp, [[11.25, 13.5], [20.25, 22.5]])

    def test_simple_input_conv_relu(self):
        net = BasicModel_SmallConvNet()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        self._grad_cam_test_assert(net, net.relu1, inp, [[0.0, 4.0], [28.0, 32.5]])

    def test_simple_multi_input_conv(self):
        net = BasicModel_SmallConvNet()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        self._grad_cam_test_assert(
            net, net.conv1, (inp, inp2), [[14.5, 19.0], [32.5, 37.0]]
        )

    def _grad_cam_test_assert(
        self,
        model,
        target_layer,
        test_input,
        expected_activation,
        additional_input=None,
    ):
        layer_gc = LayerGradCam(model, target_layer)
        attributions = layer_gc.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        assertTensorAlmostEqual(self, attributions, expected_activation, delta=0.01)


if __name__ == "__main__":
    unittest.main()
