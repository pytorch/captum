#!/usr/bin/env python3
import unittest

import captum.optim._utils.image.common as common
import torch
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


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

        x_out_expected = torch.tensor(
            [
                [
                    [0.9639, 0.9639, 0.9639, 0.9639],
                    [0.8580, 0.8580, 0.8580, 0.8580],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8102, 0.8102, 0.8102, 0.8102],
                    [0.2408, 0.2408, 0.2408, 0.2408],
                ],
                [
                    [0.8400, 0.8400, 0.8400, 0.8400],
                    [0.2588, 0.2588, 0.2588, 0.2588],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8902, 0.8902, 0.8902, 0.8902],
                    [0.5749, 0.5749, 0.5749, 0.5749],
                ],
                [
                    [0.7851, 0.7851, 0.7851, 0.7851],
                    [0.2792, 0.2792, 0.2792, 0.2792],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.9294, 0.9294, 0.9294, 0.9294],
                    [0.7624, 0.7624, 0.7624, 0.7624],
                ],
            ]
        )
        assertTensorAlmostEqual(self, x_out, x_out_expected, delta=0.01)

    def test_weights_to_heatmap_2d_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping weights_to_heatmap_2d CUDA test due to not supporting CUDA."
            )
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x.cuda())

        x_out_expected = torch.tensor(
            [
                [
                    [0.9639, 0.9639, 0.9639, 0.9639],
                    [0.8580, 0.8580, 0.8580, 0.8580],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8102, 0.8102, 0.8102, 0.8102],
                    [0.2408, 0.2408, 0.2408, 0.2408],
                ],
                [
                    [0.8400, 0.8400, 0.8400, 0.8400],
                    [0.2588, 0.2588, 0.2588, 0.2588],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8902, 0.8902, 0.8902, 0.8902],
                    [0.5749, 0.5749, 0.5749, 0.5749],
                ],
                [
                    [0.7851, 0.7851, 0.7851, 0.7851],
                    [0.2792, 0.2792, 0.2792, 0.2792],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.9294, 0.9294, 0.9294, 0.9294],
                    [0.7624, 0.7624, 0.7624, 0.7624],
                ],
            ]
        )
        assertTensorAlmostEqual(self, x_out, x_out_expected, delta=0.01)
        self.assertTrue(x_out.is_cuda)
