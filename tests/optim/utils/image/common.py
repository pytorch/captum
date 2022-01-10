#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.image.common as common
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.optim.helpers import numpy_common


class TestMakeGridImage(BaseTest):
    def test_make_grid_image_single_tensor(self) -> None:
        test_input = torch.ones(6, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, nrow=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_tensor_list(self) -> None:
        test_input = [torch.ones(1, 1, 2, 2) for i in range(6)]
        test_output = common.make_grid_image(
            test_input, nrow=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_fewer_tiles(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, nrow=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_padding(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, nrow=2, padding=2, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 10, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, nrow=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping make_image_grid CUDA test due to not supporting" + " CUDA."
            )
        test_input = torch.ones(4, 1, 2, 2).cuda()
        test_output = common.make_grid_image(
            test_input, nrow=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        self.assertTrue(test_output.is_cuda)
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping make_image_grid JIT module test due to"
                + "  insufficient Torch version."
            )
        test_input = torch.ones(4, 1, 2, 2)
        jit_make_grid_image = torch.jit.script(common.make_grid_image)
        test_output = jit_make_grid_image(test_input, nrow=2, padding=1, pad_value=5.0)
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)


class TestGetNeuronPos(BaseTest):
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
