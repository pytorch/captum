from __future__ import print_function

import unittest

import torch
from captum.attr._core.guided_grad_cam import GuidedGradCam

from .helpers.basic_models import BasicModel_ConvNet_One_Conv
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input_conv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        ex = [
            [0.0, 0.0, 4.0, 4.0],
            [0.0, 0.0, 12.0, 8.0],
            [28.0, 84.0, 97.5, 65.0],
            [28.0, 56.0, 65.0, 32.5],
        ]
        self._guided_grad_cam_test_assert(net, net.relu1, (inp,), (ex,))

    def test_simple_multi_input_conv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        ex = [
            [14.5, 29.0, 38.0, 19.0],
            [29.0, 58.0, 76.0, 38.0],
            [65.0, 130.0, 148.0, 74.0],
            [32.5, 65.0, 74.0, 37.0],
        ]
        self._guided_grad_cam_test_assert(
            net, net.conv1, (inp, inp2), ex, chosen_input_index=0
        )
        self._guided_grad_cam_test_assert(
            net, net.conv1, (inp, inp2), ex, chosen_input_index=1
        )

    def _guided_grad_cam_test_assert(
        self,
        model,
        target_layer,
        test_input,
        expected,
        additional_input=None,
        chosen_input_index=0,
    ):
        guided_gc = GuidedGradCam(model, target_layer)
        attributions = guided_gc.attribute(
            test_input,
            target=0,
            additional_forward_args=additional_input,
            chosen_input_index=chosen_input_index,
        )
        assertTensorAlmostEqual(self, attributions, expected, delta=0.01)


if __name__ == "__main__":
    unittest.main()
