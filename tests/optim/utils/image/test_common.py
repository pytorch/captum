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


class TestDotCossim(BaseTest):
    def test_dot_cossim_cossim_pow_0(self) -> None:
        x = torch.arange(0, 1 * 3 * 4 * 4).view(1, 3, 4, 4).float()
        y = torch.roll(x.clone(), shifts=(1, 2, 2, 1), dims=(0, 1, 2, 3))
        test_output = common._dot_cossim(x, y, cossim_pow=0.0)

        expected_output = torch.tensor(
            [
                [
                    [1040.0, 968.0, 1094.0, 1226.0],
                    [1604.0, 1508.0, 1658.0, 1814.0],
                    [1112.0, 944.0, 1070.0, 1202.0],
                    [1676.0, 1484.0, 1634.0, 1790.0],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output)

    def test_dot_cossim_cossim_pow_4(self) -> None:
        x = torch.arange(0, 1 * 3 * 4 * 4).view(1, 3, 4, 4).float()
        y = torch.roll(x.clone(), shifts=(1, 2, 2, 1), dims=(0, 1, 2, 3))
        test_output = common._dot_cossim(x, y, cossim_pow=4.0)

        expected_output = torch.tensor(
            [
                [
                    [101.9391, 89.0743, 124.8861, 168.7577],
                    [314.2930, 282.3505, 352.6324, 432.1260],
                    [133.2007, 80.3036, 114.3202, 156.4043],
                    [365.9309, 266.5905, 335.3027, 413.3354],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.0005)


class TestNChannelsToRGB(BaseTest):
    def test_nchannels_to_rgb_collapse(self) -> None:
        test_input = torch.arange(0, 1 * 4 * 4 * 4).view(1, 4, 4, 4).float()
        test_output = common.nchannels_to_rgb(test_input, warp=True)
        expected_output = torch.tensor(
            [
                [
                    [
                        [31.6934, 32.6204, 33.5554, 34.4981],
                        [35.4482, 36.4053, 37.3690, 38.3390],
                        [39.3149, 40.2964, 41.2832, 42.2750],
                        [43.2715, 44.2725, 45.2776, 46.2866],
                    ],
                    [
                        [20.6687, 21.5674, 22.4618, 23.3529],
                        [24.2417, 25.1290, 26.0154, 26.9013],
                        [27.7870, 28.6729, 29.5592, 30.4460],
                        [31.3335, 32.2217, 33.1109, 34.0009],
                    ],
                    [
                        [46.3932, 47.4421, 48.5129, 49.6036],
                        [50.7125, 51.8380, 52.9788, 54.1335],
                        [55.3011, 56.4806, 57.6710, 58.8715],
                        [60.0815, 61.3001, 62.5268, 63.7611],
                    ],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.005)

    def test_nchannels_to_rgb_collapse_warp_false(self) -> None:
        test_input = torch.arange(0, 1 * 4 * 4 * 4).view(1, 4, 4, 4).float()
        test_output = common.nchannels_to_rgb(test_input, warp=False)
        expected_output = torch.tensor(
            [
                [
                    [
                        [28.4279, 29.3496, 30.2753, 31.2053],
                        [32.1396, 33.0782, 34.0210, 34.9679],
                        [35.9188, 36.8736, 37.8322, 38.7943],
                        [39.7598, 40.7286, 41.7006, 42.6756],
                    ],
                    [
                        [20.5599, 21.4595, 22.3544, 23.2459],
                        [24.1351, 25.0225, 25.9088, 26.7946],
                        [27.6801, 28.5657, 29.4515, 30.3378],
                        [31.2247, 32.1124, 33.0008, 33.8900],
                    ],
                    [
                        [48.5092, 49.5791, 50.6723, 51.7866],
                        [52.9201, 54.0713, 55.2386, 56.4206],
                        [57.6164, 58.8246, 60.0444, 61.2749],
                        [62.5153, 63.7649, 65.0231, 66.2892],
                    ],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.005)

    def test_nchannels_to_rgb_increase(self) -> None:
        test_input = torch.arange(0, 1 * 2 * 4 * 4).view(1, 2, 4, 4).float()
        test_output = common.nchannels_to_rgb(test_input, warp=True)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.9234, 1.7311, 2.4623],
                        [3.1419, 3.7855, 4.4036, 5.0033],
                        [5.5894, 6.1654, 6.7337, 7.2961],
                        [7.8540, 8.4083, 8.9597, 9.5089],
                    ],
                    [
                        [11.3136, 12.0238, 12.7476, 13.4895],
                        [14.2500, 15.0278, 15.8210, 16.6277],
                        [17.4464, 18.2754, 19.1135, 19.9595],
                        [20.8124, 21.6714, 22.5357, 23.4049],
                    ],
                    [
                        [11.3136, 12.0238, 12.7476, 13.4895],
                        [14.2500, 15.0278, 15.8210, 16.6277],
                        [17.4464, 18.2754, 19.1135, 19.9595],
                        [20.8124, 21.6714, 22.5357, 23.4049],
                    ],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.005)

    def test_nchannels_to_rgb_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping nchannels_to_rgb CUDA test due to not supporting CUDA."
            )
        test_input = torch.randn(1, 6, 224, 224).cuda()
        test_output = common.nchannels_to_rgb(test_input)
        self.assertTrue(test_output.is_cuda)
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

    def test_weights_to_heatmap_2d_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping ImageTensor CUDA test due to not supporting CUDA."
            )
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x.cuda())
        x_out_np = numpy_common.weights_to_heatmap_2d(x.numpy())
        assertTensorAlmostEqual(self, x_out, torch.as_tensor(x_out_np).float())
        self.assertTrue(x_out.is_cuda)
