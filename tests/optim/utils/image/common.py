#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.image.common as common
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.optim.helpers import numpy_common


class TestGetNeuronPos(unittest.TestCase):
    def test_get_neuron_pos_hw(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, H // 2)

    def test_get_neuron_pos_xy(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, 5, 5)

        self.assertEqual(x, 5)
        self.assertEqual(y, 5)

    def test_get_neuron_pos_x_none(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, 5, None)

        self.assertEqual(x, 5)
        self.assertEqual(y, H // 2)

    def test_get_neuron_pos_none_y(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, None, 5)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, 5)


class TestNChannelsToRGB(BaseTest):
    def test_nchannels_to_rgb_collapse(self) -> None:
        test_input = torch.randn(1, 6, 224, 224)
        test_output = common.nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])

    def test_nchannels_to_rgb_increase(self) -> None:
        test_input = torch.randn(1, 2, 224, 224)
        test_output = common.nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])


class TestWeightsToHeatmap2D(BaseTest):
    def test_weights_to_heatmap_2d(self) -> None:
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x)
        x_out_np = numpy_common.weights_to_heatmap_2d(x.numpy())
        assertTensorAlmostEqual(self, x_out, torch.as_tensor(x_out_np).float())


if __name__ == "__main__":
    unittest.main()
