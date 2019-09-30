from __future__ import print_function

import unittest

import torch
from captum.attr._core.grad_cam import LayerGradCam

from .helpers.basic_models import TestModel_MultiLayer, TestModel_MultiLayer_MultiInput
from .helpers.utils import assertArraysAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_non_conv(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._grad_cam_test_assert(net, net.linear0, inp, [400.0])

    def test_simple_non_conv_multi_input_linear2(self):
        net = TestModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._grad_cam_test_assert(
            net, net.model.linear2, (inp1, inp2, inp3), [392.0], (4,)
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
        print(attributions)
        assertArraysAlmostEqual(
            attributions.squeeze(0).tolist(), expected_activation, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
