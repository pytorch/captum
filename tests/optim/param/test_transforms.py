#!/usr/bin/env python3
import unittest
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import captum.optim._param.image.transforms as transforms
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.optim.helpers import numpy_transforms


class TestRandomScale(BaseTest):
    def test_random_scale_init(self) -> None:
        scale_module = transforms.RandomScale(scale=[1, 0.975, 1.025, 0.95, 1.05])
        self.assertEqual(scale_module.scale, [1.0, 0.975, 1.025, 0.95, 1.05])
        self.assertFalse(scale_module._is_distribution)
        self.assertEqual(scale_module.mode, "bilinear")
        self.assertFalse(scale_module.align_corners)
        self.assertFalse(scale_module.recompute_scale_factor)

    def test_random_scale_tensor_scale(self) -> None:
        scale = torch.tensor([1, 0.975, 1.025, 0.95, 1.05])
        scale_module = transforms.RandomScale(scale=scale)
        self.assertEqual(scale_module.scale, scale.tolist())

    def test_random_scale_int_scale(self) -> None:
        scale = [1, 2, 3, 4, 5]
        scale_module = transforms.RandomScale(scale=scale)
        for s in scale_module.scale:
            self.assertIsInstance(s, float)
        self.assertEqual(scale_module.scale, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_random_scale_scale_distributions(self) -> None:
        scale = torch.distributions.Uniform(0.95, 1.05)
        scale_module = transforms.RandomScale(scale=scale)
        self.assertIsInstance(
            scale_module.scale_distribution,
            torch.distributions.distribution.Distribution,
        )
        self.assertTrue(scale_module._is_distribution)

    def test_random_scale_torch_version_check(self) -> None:
        scale_module = transforms.RandomScale([1.0])

        has_align_corners = torch.__version__ >= "1.3.0"
        self.assertEqual(scale_module._has_align_corners, has_align_corners)

        has_recompute_scale_factor = torch.__version__ >= "1.6.0"
        self.assertEqual(
            scale_module._has_recompute_scale_factor, has_recompute_scale_factor
        )

    def test_random_scale_downscaling(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5])
        test_tensor = torch.arange(0, 1 * 1 * 10 * 10).view(1, 1, 10, 10).float()

        scaled_tensor = scale_module._scale_tensor(test_tensor, 0.5)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [5.5000, 7.5000, 9.5000, 11.5000, 13.5000],
                        [25.5000, 27.5000, 29.5000, 31.5000, 33.5000],
                        [45.5000, 47.5000, 49.5000, 51.5000, 53.5000],
                        [65.5000, 67.5000, 69.5000, 71.5000, 73.5000],
                        [85.5000, 87.5000, 89.5000, 91.5000, 93.5000],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_scale_upscaling(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5])
        test_tensor = torch.arange(0, 1 * 1 * 2 * 2).view(1, 1, 2, 2).float()

        scaled_tensor = scale_module._scale_tensor(test_tensor, 1.5)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.5000, 1.0000],
                        [1.0000, 1.5000, 2.0000],
                        [2.0000, 2.5000, 3.0000],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_forward_exact(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5])
        test_tensor = torch.arange(0, 1 * 1 * 10 * 10).view(1, 1, 10, 10).float()

        scaled_tensor = scale_module(test_tensor)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [5.5000, 7.5000, 9.5000, 11.5000, 13.5000],
                        [25.5000, 27.5000, 29.5000, 31.5000, 33.5000],
                        [45.5000, 47.5000, 49.5000, 51.5000, 53.5000],
                        [65.5000, 67.5000, 69.5000, 71.5000, 73.5000],
                        [85.5000, 87.5000, 89.5000, 91.5000, 93.5000],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_scale_forward_exact_nearest(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5], mode="nearest")
        self.assertIsNone(scale_module.align_corners)
        self.assertEqual(scale_module.mode, "nearest")

        test_tensor = torch.arange(0, 1 * 1 * 10 * 10).view(1, 1, 10, 10).float()

        scaled_tensor = scale_module(test_tensor)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [0.0, 2.0, 4.0, 6.0, 8.0],
                        [20.0, 22.0, 24.0, 26.0, 28.0],
                        [40.0, 42.0, 44.0, 46.0, 48.0],
                        [60.0, 62.0, 64.0, 66.0, 68.0],
                        [80.0, 82.0, 84.0, 86.0, 88.0],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_scale_forward_exact_align_corners(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5], align_corners=True)
        self.assertTrue(scale_module.align_corners)

        test_tensor = torch.arange(0, 1 * 1 * 10 * 10).view(1, 1, 10, 10).float()

        scaled_tensor = scale_module(test_tensor)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [0.0000, 2.2500, 4.5000, 6.7500, 9.0000],
                        [22.5000, 24.7500, 27.0000, 29.2500, 31.5000],
                        [45.0000, 47.2500, 49.5000, 51.7500, 54.0000],
                        [67.5000, 69.7500, 72.0000, 74.2500, 76.5000],
                        [90.0000, 92.2500, 94.5000, 96.7500, 99.0000],
                    ]
                ]
            ]
        )
        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_scale_forward(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5])
        test_tensor = torch.ones(1, 3, 10, 10)
        output_tensor = scale_module(test_tensor)
        self.assertEqual(list(output_tensor.shape), [1, 3, 5, 5])

    def test_random_scale_forward_distributions(self) -> None:
        scale = torch.distributions.Uniform(0.95, 1.05)
        scale_module = transforms.RandomScale(scale=scale)
        test_tensor = torch.ones(1, 3, 10, 10)
        output_tensor = scale_module(test_tensor)
        self.assertTrue(torch.is_tensor(output_tensor))

    def test_random_scale_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping RandomScale JIT module test due to insufficient"
                + " Torch version."
            )
        scale_module = transforms.RandomScale(scale=[1.5])
        jit_scale_module = torch.jit.script(scale_module)

        test_tensor = torch.arange(0, 1 * 1 * 2 * 2).view(1, 1, 2, 2).float()
        scaled_tensor = jit_scale_module(test_tensor)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.5000, 1.0000],
                        [1.0000, 1.5000, 2.0000],
                        [2.0000, 2.5000, 3.0000],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )


class TestRandomScaleAffine(BaseTest):
    def test_random_scale_affine_init(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[1, 0.975, 1.025, 0.95, 1.05])
        self.assertEqual(scale_module.scale, [1.0, 0.975, 1.025, 0.95, 1.05])
        self.assertFalse(scale_module._is_distribution)
        self.assertEqual(scale_module.mode, "bilinear")
        self.assertEqual(scale_module.padding_mode, "zeros")
        self.assertFalse(scale_module.align_corners)

    def test_random_scale_affine_tensor_scale(self) -> None:
        scale = torch.tensor([1, 0.975, 1.025, 0.95, 1.05])
        scale_module = transforms.RandomScaleAffine(scale=scale)
        self.assertEqual(scale_module.scale, scale.tolist())

    def test_random_scale_affine_int_scale(self) -> None:
        scale = [1, 2, 3, 4, 5]
        scale_module = transforms.RandomScaleAffine(scale=scale)
        for s in scale_module.scale:
            self.assertIsInstance(s, float)
        self.assertEqual(scale_module.scale, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_random_scale_affine_scale_distributions(self) -> None:
        scale = torch.distributions.Uniform(0.95, 1.05)
        scale_module = transforms.RandomScaleAffine(scale=scale)
        self.assertIsInstance(
            scale_module.scale_distribution,
            torch.distributions.distribution.Distribution,
        )
        self.assertTrue(scale_module._is_distribution)

    def test_random_scale_affine_torch_version_check(self) -> None:
        scale_module = transforms.RandomScaleAffine([1.0])
        _has_align_corners = torch.__version__ >= "1.3.0"
        self.assertEqual(scale_module._has_align_corners, _has_align_corners)

    def test_random_scale_affine_matrix(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[0.5])
        test_tensor = torch.ones(1, 3, 3, 3)
        # Test scale matrices

        assertTensorAlmostEqual(
            self,
            scale_module._get_scale_mat(0.5, test_tensor.device, test_tensor.dtype),
            torch.tensor([[0.5000, 0.0000, 0.0000], [0.0000, 0.5000, 0.0000]]),
            0,
        )

        assertTensorAlmostEqual(
            self,
            scale_module._get_scale_mat(1.24, test_tensor.device, test_tensor.dtype),
            torch.tensor([[1.2400, 0.0000, 0.0000], [0.0000, 1.2400, 0.0000]]),
            0,
        )

    def test_random_scale_affine_downscaling(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[0.5])
        test_tensor = torch.ones(1, 3, 3, 3)

        assertTensorAlmostEqual(
            self,
            scale_module._scale_tensor(test_tensor, 0.5),
            torch.ones(3, 1).repeat(3, 1, 3).unsqueeze(0),
            0,
        )

    def test_random_scale_affine_upscaling(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[1.5])
        test_tensor = torch.ones(1, 3, 3, 3)

        assertTensorAlmostEqual(
            self,
            scale_module._scale_tensor(test_tensor, 1.5),
            torch.tensor(
                [
                    [0.2500, 0.5000, 0.2500],
                    [0.5000, 1.0000, 0.5000],
                    [0.2500, 0.5000, 0.2500],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )

    def test_random_scale_affine_forward_exact(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[1.5])
        test_tensor = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()

        scaled_tensor = scale_module(test_tensor)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.1875, 0.5625, 0.1875],
                        [0.7500, 3.7500, 5.2500, 1.5000],
                        [2.2500, 9.7500, 11.2500, 3.0000],
                        [0.7500, 3.1875, 3.5625, 0.9375],
                    ]
                ]
            ]
        )
        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_scale_affine_forward_exact_mode_nearest(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[1.5], mode="nearest")
        self.assertEqual(scale_module.mode, "nearest")
        test_tensor = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()

        scaled_tensor = scale_module(test_tensor)
        expected_tensor = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 5.0, 6.0, 0.0],
                        [0.0, 9.0, 10.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0,
        )

    def test_random_scale_affine_forward(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[0.5])
        test_tensor = torch.ones(1, 3, 10, 10)
        output_tensor = scale_module(test_tensor)
        self.assertEqual(list(output_tensor.shape), list(test_tensor.shape))

    def test_random_scale_affine_forward_distributions(self) -> None:
        scale = torch.distributions.Uniform(0.95, 1.05)
        scale_module = transforms.RandomScaleAffine(scale=scale)
        test_tensor = torch.ones(1, 3, 10, 10)
        output_tensor = scale_module(test_tensor)
        self.assertEqual(list(output_tensor.shape), list(test_tensor.shape))

    def test_random_scale_affine_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping RandomScaleAffine JIT module test due to insufficient"
                + " Torch version."
            )
        scale_module = transforms.RandomScaleAffine(scale=[1.5])
        jit_scale_module = torch.jit.script(scale_module)
        test_tensor = torch.ones(1, 3, 3, 3)

        assertTensorAlmostEqual(
            self,
            jit_scale_module(test_tensor),
            torch.tensor(
                [
                    [0.2500, 0.5000, 0.2500],
                    [0.5000, 1.0000, 0.5000],
                    [0.2500, 0.5000, 0.2500],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )


class TestRandomSpatialJitter(BaseTest):
    def test_random_spatial_jitter_init(self) -> None:
        translate = 3
        spatialjitter = transforms.RandomSpatialJitter(translate)

        self.assertEqual(spatialjitter.pad_range, translate * 2)
        self.assertIsInstance(spatialjitter.pad, torch.nn.ReflectionPad2d)

    def test_random_spatial_jitter_hw(self) -> None:
        translate_vals = [4, 4]
        t_val = 3

        spatialjitter = transforms.RandomSpatialJitter(t_val)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        jittered_tensor = spatialjitter.translate_tensor(
            test_input, torch.tensor(translate_vals)
        ).squeeze(0)

        spatial_mod_np = numpy_transforms.RandomSpatialJitter(t_val)
        jittered_array = spatial_mod_np.translate_array(np.eye(4, 4), translate_vals)

        assertArraysAlmostEqual(jittered_tensor[0].numpy(), jittered_array, 0)
        assertArraysAlmostEqual(jittered_tensor[1].numpy(), jittered_array, 0)
        assertArraysAlmostEqual(jittered_tensor[2].numpy(), jittered_array, 0)

    def test_random_spatial_jitter_w(self) -> None:
        translate_vals = [0, 3]
        t_val = 3

        spatialjitter = transforms.RandomSpatialJitter(t_val)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        jittered_tensor = spatialjitter.translate_tensor(
            test_input, torch.tensor(translate_vals)
        ).squeeze(0)

        spatial_mod_np = numpy_transforms.RandomSpatialJitter(t_val)
        jittered_array = spatial_mod_np.translate_array(np.eye(4, 4), translate_vals)

        assertArraysAlmostEqual(jittered_tensor[0].numpy(), jittered_array, 0)
        assertArraysAlmostEqual(jittered_tensor[1].numpy(), jittered_array, 0)
        assertArraysAlmostEqual(jittered_tensor[2].numpy(), jittered_array, 0)

    def test_random_spatial_jitter_h(self) -> None:
        translate_vals = [3, 0]
        t_val = 3

        spatialjitter = transforms.RandomSpatialJitter(t_val)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        jittered_tensor = spatialjitter.translate_tensor(
            test_input, torch.tensor(translate_vals)
        ).squeeze(0)

        spatial_mod_np = numpy_transforms.RandomSpatialJitter(t_val)
        jittered_array = spatial_mod_np.translate_array(np.eye(4, 4), translate_vals)

        assertArraysAlmostEqual(jittered_tensor[0].numpy(), jittered_array, 0)
        assertArraysAlmostEqual(jittered_tensor[1].numpy(), jittered_array, 0)
        assertArraysAlmostEqual(jittered_tensor[2].numpy(), jittered_array, 0)

    def test_random_spatial_jitter_forward(self) -> None:
        t_val = 3

        spatialjitter = transforms.RandomSpatialJitter(t_val)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        jittered_tensor = spatialjitter(test_input)
        self.assertEqual(list(jittered_tensor.shape), list(test_input.shape))

    def test_random_spatial_jitter_forward_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping RandomSpatialJitter JIT module test due to insufficient"
                + " Torch version."
            )
        t_val = 3

        spatialjitter = transforms.RandomSpatialJitter(t_val)
        jit_spatialjitter = torch.jit.script(spatialjitter)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        jittered_tensor = jit_spatialjitter(test_input)
        self.assertEqual(list(jittered_tensor.shape), list(test_input.shape))


class TestCenterCrop(BaseTest):
    def test_center_crop_init(self) -> None:
        crop_module = transforms.CenterCrop(3)
        self.assertEqual(crop_module.size, [3, 3])
        self.assertFalse(crop_module.pixels_from_edges)
        self.assertFalse(crop_module.offset_left)
        self.assertEqual(crop_module.padding_mode, "constant")
        self.assertEqual(crop_module.padding_value, 0.0)

    def test_center_crop_forward_one_number(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = 3

        crop_tensor = transforms.CenterCrop(crop_vals, True)
        cropped_tensor = crop_tensor(test_tensor)

        crop_mod_np = numpy_transforms.CenterCrop(crop_vals, True)
        cropped_array = crop_mod_np.forward(test_tensor.numpy())

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_one_number_dim_3(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1).repeat(
            3, 1, 1
        )
        crop_vals = 3

        crop_tensor = transforms.CenterCrop(crop_vals, True)
        cropped_tensor = crop_tensor(test_tensor)

        crop_mod_np = numpy_transforms.CenterCrop(crop_vals, True)
        cropped_array = crop_mod_np.forward(test_tensor.numpy())

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * 3
        )
        self.assertEqual(cropped_tensor.shape, expected_tensor.shape)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_one_number_list(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = [3]

        crop_tensor = transforms.CenterCrop(crop_vals, True)
        cropped_tensor = crop_tensor(test_tensor)

        crop_mod_np = numpy_transforms.CenterCrop(crop_vals, True)
        cropped_array = crop_mod_np.forward(test_tensor.numpy())

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_list_len_3_value_error(self) -> None:
        crop_vals = [3, 3, 3]

        with self.assertRaises(ValueError):
            transforms.CenterCrop(crop_vals, True)

    def test_center_crop_str_value_error(self) -> None:
        crop_vals = "error"

        with self.assertRaises(ValueError):
            transforms.CenterCrop(crop_vals, True)

    def test_center_crop_forward_two_numbers(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = (4, 0)

        crop_tensor = transforms.CenterCrop(crop_vals, True)
        cropped_tensor = crop_tensor(test_tensor)

        crop_mod_np = numpy_transforms.CenterCrop(crop_vals, True)
        cropped_array = crop_mod_np.forward(test_tensor.numpy())

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.stack([torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])] * 2)] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_one_number_exact(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_vals = 5

        crop_tensor = transforms.CenterCrop(crop_vals, False)
        cropped_tensor = crop_tensor(test_tensor)

        crop_mod_np = numpy_transforms.CenterCrop(crop_vals, False)
        cropped_array = crop_mod_np.forward(test_tensor.numpy())

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ]
            * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_two_numbers_exact(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_vals = (4, 2)

        crop_tensor = transforms.CenterCrop(crop_vals, False)
        cropped_tensor = crop_tensor(test_tensor)

        crop_mod_np = numpy_transforms.CenterCrop(crop_vals, False)
        cropped_array = crop_mod_np.forward(test_tensor.numpy())

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)

        expected_tensor = torch.stack(
            [torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_offset_left_uneven_sides(self) -> None:
        crop_mod = transforms.CenterCrop(
            [5, 5], pixels_from_edges=False, offset_left=True
        )
        x = torch.ones(1, 3, 5, 5)
        px = F.pad(x, (5, 4, 5, 4), value=float("-inf"))
        cropped_tensor = crop_mod(px)
        assertTensorAlmostEqual(self, x, cropped_tensor)

    def test_center_crop_forward_offset_left_even_sides(self) -> None:
        crop_mod = transforms.CenterCrop(
            [5, 5], pixels_from_edges=False, offset_left=True
        )
        x = torch.ones(1, 3, 5, 5)
        px = F.pad(x, (5, 5, 5, 5), value=float("-inf"))
        cropped_tensor = crop_mod(px)
        assertTensorAlmostEqual(self, x, cropped_tensor)

    def test_center_crop_forward_padding(self) -> None:
        test_tensor = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()
        crop_vals = [6, 6]

        center_crop_module = transforms.CenterCrop(crop_vals, offset_left=False)
        cropped_tensor = center_crop_module(test_tensor)

        expected_tensor = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 6.0, 7.0, 0.0],
                [0.0, 8.0, 9.0, 10.0, 11.0, 0.0],
                [0.0, 12.0, 13.0, 14.0, 15.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_padding_prime_num_pad(self) -> None:
        test_tensor = torch.arange(0, 1 * 1 * 3 * 3).view(1, 1, 3, 3).float()
        crop_vals = [6, 6]

        center_crop_module = transforms.CenterCrop(crop_vals, offset_left=False)

        cropped_tensor = center_crop_module(test_tensor)

        expected_tensor = torch.nn.functional.pad(test_tensor, [2, 1, 2, 1])
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_forward_padding_prime_num_pad_offset_left(self) -> None:
        test_tensor = torch.arange(0, 1 * 1 * 3 * 3).view(1, 1, 3, 3).float()
        crop_vals = [6, 6]

        center_crop_module = transforms.CenterCrop(crop_vals, offset_left=True)

        cropped_tensor = center_crop_module(test_tensor)

        expected_tensor = torch.nn.functional.pad(test_tensor, [1, 2, 1, 2])
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_forward_one_number_exact_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping CenterCrop JIT module test due to insufficient"
                + " Torch version."
            )
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_vals = 5

        crop_tensor = transforms.CenterCrop(crop_vals, False)
        jit_crop_tensor = torch.jit.script(crop_tensor)
        cropped_tensor = jit_crop_tensor(test_tensor)
        expected_tensor = torch.stack(
            [
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ]
            * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_forward_padding_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping CenterCrop padding JIT module test due to insufficient"
                + " Torch version."
            )
        test_tensor = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()
        crop_vals = [6, 6]

        center_crop_module = transforms.CenterCrop(crop_vals, offset_left=False)
        jit_center_crop = torch.jit.script(center_crop_module)
        cropped_tensor = jit_center_crop(test_tensor)

        expected_tensor = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 6.0, 7.0, 0.0],
                [0.0, 8.0, 9.0, 10.0, 11.0, 0.0],
                [0.0, 12.0, 13.0, 14.0, 15.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)


class TestCenterCropFunction(BaseTest):
    def test_center_crop_one_number(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = 3

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals, True)
        cropped_array = numpy_transforms.center_crop(
            test_tensor.numpy(), crop_vals, True
        )

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_one_number_dim_3(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1).repeat(
            3, 1, 1
        )
        crop_vals = 3

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals, True)
        cropped_array = numpy_transforms.center_crop(
            test_tensor.numpy(), crop_vals, True
        )

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * 3
        )
        self.assertEqual(cropped_tensor.shape, expected_tensor.shape)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_one_number_list(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = [3]

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals, True)
        cropped_array = numpy_transforms.center_crop(
            test_tensor.numpy(), crop_vals, True
        )

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_list_len_3_value_error(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = [3, 3, 3]

        with self.assertRaises(ValueError):
            transforms.center_crop(test_tensor, crop_vals, True)

    def test_center_crop_str_value_error(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = "error"

        with self.assertRaises(ValueError):
            transforms.center_crop(test_tensor, crop_vals, True)

    def test_center_crop_two_numbers(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        crop_vals = (4, 2)

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals, True)
        cropped_array = numpy_transforms.center_crop(
            test_tensor.numpy(), crop_vals, True
        )

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.stack([torch.tensor([0.0, 1.0, 1.0, 0.0])] * 2)] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_one_number_exact(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_vals = 5

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals, False)
        cropped_array = numpy_transforms.center_crop(
            test_tensor.numpy(), crop_vals, False
        )

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ]
            * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_two_numbers_exact(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_vals = (4, 2)

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals, False)
        cropped_array = numpy_transforms.center_crop(
            test_tensor.numpy(), crop_vals, False
        )

        assertArraysAlmostEqual(cropped_tensor.numpy(), cropped_array, 0)
        expected_tensor = torch.stack(
            [torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])] * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_offset_left_uneven_sides(self) -> None:
        x = torch.ones(1, 3, 5, 5)
        px = F.pad(x, (5, 4, 5, 4), value=float("-inf"))
        cropped_tensor = transforms.center_crop(
            px, size=[5, 5], pixels_from_edges=False, offset_left=True
        )
        assertTensorAlmostEqual(self, x, cropped_tensor)

    def test_center_crop_offset_left_even_sides(self) -> None:
        x = torch.ones(1, 3, 5, 5)
        px = F.pad(x, (5, 5, 5, 5), value=float("-inf"))
        cropped_tensor = transforms.center_crop(
            px, size=[5, 5], pixels_from_edges=False, offset_left=True
        )
        assertTensorAlmostEqual(self, x, cropped_tensor)

    def test_center_crop_padding(self) -> None:
        test_tensor = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()
        crop_vals = [6, 6]

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals)

        expected_tensor = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 6.0, 7.0, 0.0],
                [0.0, 8.0, 9.0, 10.0, 11.0, 0.0],
                [0.0, 12.0, 13.0, 14.0, 15.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)

    def test_center_crop_padding_prime_num_pad(self) -> None:
        test_tensor = torch.arange(0, 1 * 1 * 3 * 3).view(1, 1, 3, 3).float()
        crop_vals = [6, 6]

        cropped_tensor = transforms.center_crop(test_tensor, crop_vals)

        expected_tensor = torch.nn.functional.pad(test_tensor, [2, 1, 2, 1])
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_padding_prime_num_pad_offset_left(self) -> None:
        test_tensor = torch.arange(0, 1 * 1 * 3 * 3).view(1, 1, 3, 3).float()
        crop_vals = [6, 6]

        cropped_tensor = transforms.center_crop(
            test_tensor, crop_vals, offset_left=True
        )

        expected_tensor = torch.nn.functional.pad(test_tensor, [1, 2, 1, 2])
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_one_number_exact_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping center_crop JIT module test due to insufficient"
                + " Torch version."
            )
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_vals = 5

        jit_center_crop = transforms.center_crop
        cropped_tensor = jit_center_crop(test_tensor, crop_vals, False)
        expected_tensor = torch.stack(
            [
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ]
            * 3
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor)

    def test_center_crop_padding_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping center_crop padding JIT module test due to insufficient"
                + " Torch version."
            )
        test_tensor = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()
        crop_vals = [6, 6]

        jit_center_crop = torch.jit.script(transforms.center_crop)
        cropped_tensor = jit_center_crop(test_tensor, crop_vals)

        expected_tensor = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 6.0, 7.0, 0.0],
                [0.0, 8.0, 9.0, 10.0, 11.0, 0.0],
                [0.0, 12.0, 13.0, 14.0, 15.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor, 0)


class TestBlendAlpha(BaseTest):
    def test_blend_alpha_init(self) -> None:
        blend_alpha = transforms.BlendAlpha(background=None)
        self.assertIsNone(blend_alpha.background)

    def test_blend_alpha(self) -> None:
        rgb_tensor = torch.ones(3, 3, 3)
        alpha_tensor = ((torch.eye(3, 3) + torch.eye(3, 3).flip(1)) / 2).repeat(1, 1, 1)
        test_tensor = torch.cat([rgb_tensor, alpha_tensor]).unsqueeze(0)

        background_tensor = torch.ones_like(rgb_tensor) * 5
        blend_alpha = transforms.BlendAlpha(background=background_tensor)
        blended_tensor = blend_alpha(test_tensor)

        rgb_array = np.ones((3, 3, 3))
        alpha_array = (np.add(np.eye(3, 3), np.flip(np.eye(3, 3), 1)) / 2)[None, :]
        test_array = np.concatenate([rgb_array, alpha_array])[None, :]

        background_array = np.ones(rgb_array.shape) * 5
        blend_alpha_np = numpy_transforms.BlendAlpha(background=background_array)
        blended_array = blend_alpha_np.blend_alpha(test_array)

        assertArraysAlmostEqual(blended_tensor.numpy(), blended_array, 0)

    def test_blend_alpha_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping BlendAlpha JIT module test due to insufficient"
                + " Torch version."
            )
        rgb_tensor = torch.ones(3, 3, 3)
        alpha_tensor = ((torch.eye(3, 3) + torch.eye(3, 3).flip(1)) / 2).repeat(1, 1, 1)
        test_tensor = torch.cat([rgb_tensor, alpha_tensor]).unsqueeze(0)

        background_tensor = torch.ones_like(rgb_tensor) * 5
        blend_alpha = transforms.BlendAlpha(background=background_tensor)
        jit_blend_alpha = torch.jit.script(blend_alpha)
        blended_tensor = jit_blend_alpha(test_tensor)

        rgb_array = np.ones((3, 3, 3))
        alpha_array = (np.add(np.eye(3, 3), np.flip(np.eye(3, 3), 1)) / 2)[None, :]
        test_array = np.concatenate([rgb_array, alpha_array])[None, :]

        background_array = np.ones(rgb_array.shape) * 5
        blend_alpha_np = numpy_transforms.BlendAlpha(background=background_array)
        blended_array = blend_alpha_np.blend_alpha(test_array)

        assertArraysAlmostEqual(blended_tensor.numpy(), blended_array, 0)


class TestIgnoreAlpha(BaseTest):
    def test_ignore_alpha(self) -> None:
        ignore_alpha = transforms.IgnoreAlpha()
        test_input = torch.ones(1, 4, 3, 3)
        rgb_tensor = ignore_alpha(test_input)
        assert rgb_tensor.size(1) == 3

    def test_ignore_alpha_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping IgnoreAlpha JIT module test due to insufficient"
                + " Torch version."
            )
        ignore_alpha = transforms.IgnoreAlpha()
        jit_ignore_alpha = torch.jit.script(ignore_alpha)
        test_input = torch.ones(1, 4, 3, 3)
        rgb_tensor = jit_ignore_alpha(test_input)
        assert rgb_tensor.size(1) == 3


class TestToRGB(BaseTest):
    def test_to_rgb_init(self) -> None:
        to_rgb = transforms.ToRGB()
        self.assertEqual(list(to_rgb.transform.shape), [3, 3])
        transform = torch.tensor(
            [
                [0.5628, 0.1948, 0.0433],
                [0.5845, 0.0000, -0.1082],
                [0.5845, -0.1948, 0.0649],
            ]
        )
        assertTensorAlmostEqual(self, to_rgb.transform, transform, 0.001)

    def test_to_rgb_i1i2i3(self) -> None:
        to_rgb = transforms.ToRGB(transform="i1i2i3")
        to_rgb_np = numpy_transforms.ToRGB(transform="i1i2i3")
        assertArraysAlmostEqual(to_rgb.transform.numpy(), to_rgb_np.transform)
        transform = torch.tensor(
            [
                [0.3333, 0.3333, 0.3333],
                [0.5000, 0.0000, -0.5000],
                [-0.2500, 0.5000, -0.2500],
            ]
        )
        assertTensorAlmostEqual(self, to_rgb.transform, transform, 0.001)

    def test_to_rgb_klt(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        to_rgb_np = numpy_transforms.ToRGB(transform="klt")
        assertArraysAlmostEqual(to_rgb.transform.numpy(), to_rgb_np.transform)
        transform = torch.tensor(
            [
                [0.5628, 0.1948, 0.0433],
                [0.5845, 0.0000, -0.1082],
                [0.5845, -0.1948, 0.0649],
            ]
        )
        assertTensorAlmostEqual(self, to_rgb.transform, transform, 0.001)

    def test_to_rgb_custom(self) -> None:
        matrix = torch.eye(3, 3)
        to_rgb = transforms.ToRGB(transform=matrix)
        to_rgb_np = numpy_transforms.ToRGB(transform=matrix.numpy())
        assertArraysAlmostEqual(to_rgb.transform.numpy(), to_rgb_np.transform)
        assertTensorAlmostEqual(self, to_rgb.transform, matrix, 0.0)

    def test_to_rgb_init_value_error(self) -> None:
        with self.assertRaises(ValueError):
            transforms.ToRGB(transform="error")

    def test_to_rgb_klt_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB forward due to insufficient Torch version."
            )
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(3, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        expected_rgb_tensor = torch.stack([r, g, b]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_alpha_klt_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB with Alpha forward due to insufficient Torch version."
            )
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(4, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        a = torch.ones(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b, a]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_alpha_klt_forward_dim_3(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB with Alpha forward dim 3 due to"
                + " insufficient Torch version."
            )
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(4, 4, 4).refine_names("C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        a = torch.ones(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b, a])

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_i1i2i3_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB forward due to insufficient Torch version."
            )
        to_rgb = transforms.ToRGB(transform="i1i2i3")
        test_tensor = torch.ones(3, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4)
        g = torch.zeros(4, 4)
        b = torch.zeros(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_alpha_i1i2i3_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB with Alpha forward due to insufficient Torch version."
            )
        to_rgb = transforms.ToRGB(transform="i1i2i3")
        test_tensor = torch.ones(4, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4)
        g = torch.zeros(4, 4)
        b = torch.zeros(4, 4)
        a = torch.ones(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b, a]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_custom_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB forward due to insufficient Torch version."
            )
        matrix = torch.eye(3, 3)
        to_rgb = transforms.ToRGB(transform=matrix)
        test_tensor = torch.ones(3, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        to_rgb_np = numpy_transforms.ToRGB(transform=matrix.numpy())
        test_array = np.ones((1, 3, 4, 4))
        rgb_array = to_rgb_np.to_rgb(test_array)

        assertArraysAlmostEqual(rgb_tensor.numpy(), rgb_array)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_klt_forward_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping ToRGB forward JIT module test due to insufficient"
                + " Torch version."
            )
        to_rgb = transforms.ToRGB(transform="klt")
        jit_to_rgb = torch.jit.script(to_rgb)
        test_tensor = torch.ones(3, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")
        rgb_tensor = jit_to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        expected_rgb_tensor = torch.stack([r, g, b]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)

        inverse_tensor = jit_to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )


class TestGaussianSmoothing(BaseTest):
    def test_gaussian_smoothing_init_1d(self) -> None:
        channels = 6
        kernel_size = 3
        sigma = 2.0
        dim = 1
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )
        self.assertEqual(smoothening_module.groups, channels)
        weight = torch.tensor([[0.3192, 0.3617, 0.3192]]).repeat(6, 1, 1)
        assertTensorAlmostEqual(self, smoothening_module.weight, weight, 0.001)

    def test_gaussian_smoothing_init_2d(self) -> None:
        channels = 3
        kernel_size = 3
        sigma = 2.0
        dim = 2
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )
        self.assertEqual(smoothening_module.groups, channels)
        weight = torch.tensor(
            [
                [
                    [0.1019, 0.1154, 0.1019],
                    [0.1154, 0.1308, 0.1154],
                    [0.1019, 0.1154, 0.1019],
                ]
            ]
        ).repeat(3, 1, 1, 1)
        assertTensorAlmostEqual(self, smoothening_module.weight, weight, 0.001)

    def test_gaussian_smoothing_init_3d(self) -> None:
        channels = 4
        kernel_size = 3
        sigma = 1.021
        dim = 3
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )
        self.assertEqual(smoothening_module.groups, channels)
        weight = torch.tensor(
            [
                [
                    [
                        [0.0212, 0.0342, 0.0212],
                        [0.0342, 0.0552, 0.0342],
                        [0.0212, 0.0342, 0.0212],
                    ],
                    [
                        [0.0342, 0.0552, 0.0342],
                        [0.0552, 0.0892, 0.0552],
                        [0.0342, 0.0552, 0.0342],
                    ],
                    [
                        [0.0212, 0.0342, 0.0212],
                        [0.0342, 0.0552, 0.0342],
                        [0.0212, 0.0342, 0.0212],
                    ],
                ]
            ]
        ).repeat(4, 1, 1, 1, 1)
        assertTensorAlmostEqual(self, smoothening_module.weight, weight, 0.01)

    def test_gaussian_smoothing_init_dim_4_runtime_error(self) -> None:
        channels = 3
        kernel_size = 3
        sigma = 2.0
        dim = 4
        with self.assertRaises(RuntimeError):
            transforms.GaussianSmoothing(channels, kernel_size, sigma, dim)

    def test_gaussian_smoothing_1d(self) -> None:
        channels = 6
        kernel_size = 3
        sigma = 2.0
        dim = 1
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0]).repeat(6, 2).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(6, 1).unsqueeze(0)
        self.assertLessEqual(diff_tensor.max().item(), 4.268e-05)
        self.assertGreaterEqual(diff_tensor.min().item(), -4.197e-05)

    def test_gaussian_smoothing_2d(self) -> None:
        channels = 3
        kernel_size = 3
        sigma = 2.0
        dim = 2
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0]).repeat(3, 6, 3).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(3, 4, 2).unsqueeze(0)
        self.assertLessEqual(diff_tensor.max().item(), 4.5539e-05)
        self.assertGreaterEqual(diff_tensor.min().item(), -4.5539e-05)

    def test_gaussian_smoothing_3d(self) -> None:
        channels = 4
        kernel_size = 3
        sigma = 1.021
        dim = 3
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0, 1.0]).repeat(4, 6, 6, 2).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.7873, 2.1063, 2.1063, 2.7873]
        ).repeat(4, 4, 4, 1).unsqueeze(0)
        t_max = diff_tensor.max().item()
        t_min = diff_tensor.min().item()
        self.assertLessEqual(t_max, 4.8162e-05)
        self.assertGreaterEqual(t_min, 3.3377e-06)

    def test_gaussian_smoothing_2d_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping GaussianSmoothing 2D JIT module test due to insufficient"
                + " Torch version."
            )
        channels = 3
        kernel_size = 3
        sigma = 2.0
        dim = 2
        smoothening_module = transforms.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )
        jit_smoothening_module = torch.jit.script(smoothening_module)

        test_tensor = torch.tensor([1.0, 5.0]).repeat(3, 6, 3).unsqueeze(0)

        diff_tensor = jit_smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(3, 4, 2).unsqueeze(0)
        self.assertLessEqual(diff_tensor.max().item(), 4.5539e-05)
        self.assertGreaterEqual(diff_tensor.min().item(), -4.5539e-05)


class TestScaleInputRange(BaseTest):
    def test_scale_input_range_init(self) -> None:
        scale_input = transforms.ScaleInputRange(255)
        self.assertEqual(scale_input.multiplier, 255)

    def test_scale_input_range(self) -> None:
        x = torch.ones(1, 3, 4, 4)
        scale_input = transforms.ScaleInputRange(255)
        output_tensor = scale_input(x)
        self.assertEqual(output_tensor.mean(), 255.0)

    def test_scale_input_range_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping ScaleInputRange JIT module test due to insufficient"
                + " Torch version."
            )
        x = torch.ones(1, 3, 4, 4)
        scale_input = transforms.ScaleInputRange(255)
        jit_scale_input = torch.jit.script(scale_input)
        output_tensor = jit_scale_input(x)
        self.assertEqual(output_tensor.mean(), 255.0)


class TestRGBToBGR(BaseTest):
    def test_rgb_to_bgr(self) -> None:
        x = torch.randn(1, 3, 224, 224)
        rgb_to_bgr = transforms.RGBToBGR()
        output_tensor = rgb_to_bgr(x)
        expected_x = x[:, [2, 1, 0]]
        assertTensorAlmostEqual(self, output_tensor, expected_x)

    def test_rgb_to_bgr_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping RGBToBGR JIT module test due to insufficient"
                + " Torch version."
            )
        x = torch.randn(1, 3, 224, 224)
        rgb_to_bgr = transforms.RGBToBGR()
        jit_rgb_to_bgr = torch.jit.script(rgb_to_bgr)
        output_tensor = jit_rgb_to_bgr(x)
        expected_x = x[:, [2, 1, 0]]
        assertTensorAlmostEqual(self, output_tensor, expected_x)


class TestSymmetricPadding(BaseTest):
    def test_symmetric_padding(self) -> None:
        b = 2
        c = 3
        x = torch.arange(0, b * c * 4 * 4).view(b, c, 4, 4).float()
        offset_pad = [[3, 3], [4, 4], [2, 2], [5, 5]]

        x_pt = torch.nn.Parameter(x)
        x_out = transforms.SymmetricPadding.apply(x_pt, offset_pad)
        x_out_np = torch.as_tensor(
            np.pad(x.detach().numpy(), pad_width=offset_pad, mode="symmetric")
        )
        assertTensorAlmostEqual(self, x_out, x_out_np)

    def test_symmetric_padding_backward(self) -> None:
        b = 2
        c = 3
        x = torch.arange(0, b * c * 4 * 4).view(b, c, 4, 4).float()
        offset_pad = [[3, 3], [4, 4], [2, 2], [5, 5]]

        x_pt = torch.nn.Parameter(x) * 1

        t_grad_input, t_grad_output = [], []

        def check_grad(self, grad_input, grad_output):
            t_grad_input.append(grad_input[0].clone().detach())
            t_grad_output.append(grad_output[0].clone().detach())

        class SymmetricPaddingLayer(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, padding: List[List[int]]
            ) -> torch.Tensor:
                return transforms.SymmetricPadding.apply(x_pt, padding)

        sym_pad = SymmetricPaddingLayer()
        sym_pad.register_backward_hook(check_grad)
        x_out = sym_pad(x_pt, offset_pad)
        (x_out.sum() * 1).backward()

        self.assertEqual(x.shape, t_grad_input[0].shape)

        x_out_np = torch.as_tensor(
            np.pad(x.detach().numpy(), pad_width=offset_pad, mode="symmetric")
        )
        self.assertEqual(x_out_np.shape, t_grad_output[0].shape)


class TestNChannelsToRGB(BaseTest):
    def test_nchannels_to_rgb_init(self) -> None:
        nchannels_to_rgb = transforms.NChannelsToRGB()
        self.assertFalse(nchannels_to_rgb.warp)

    def test_nchannels_to_rgb_collapse(self) -> None:
        test_input = torch.randn(1, 6, 224, 224)
        nchannels_to_rgb = transforms.NChannelsToRGB()
        test_output = nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])

    def test_nchannels_to_rgb_increase(self) -> None:
        test_input = torch.randn(1, 2, 224, 224)
        nchannels_to_rgb = transforms.NChannelsToRGB()
        test_output = nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])

    def test_nchannels_to_rgb_collapse_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping NChannelsToRGB collapse JIT module test due to insufficient"
                + " Torch version."
            )
        test_input = torch.randn(1, 6, 224, 224)
        nchannels_to_rgb = transforms.NChannelsToRGB()
        jit_nchannels_to_rgb = torch.jit.script(nchannels_to_rgb)
        test_output = jit_nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])


class TestRandomCrop(BaseTest):
    def test_random_crop_center_crop(self) -> None:
        crop_size = [160, 160]
        crop_transform = transforms.RandomCrop(crop_size=crop_size)
        x = torch.ones(1, 4, 224, 224)

        x_out = crop_transform._center_crop(x)

        self.assertEqual(list(x_out.shape), [1, 4, 160, 160])

    def test_random_crop(self) -> None:
        crop_size = [160, 160]
        crop_transform = transforms.RandomCrop(crop_size=crop_size)
        x = torch.ones(1, 4, 224, 224)

        x_out = crop_transform(x)

        self.assertEqual(list(x_out.shape), [1, 4, 160, 160])

    def test_random_crop_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping RandomCrop JIT module test due to insufficient"
                + " Torch version."
            )
        crop_size = [160, 160]
        crop_transform = transforms.RandomCrop(crop_size=crop_size)
        jit_crop_transform = torch.jit.script(crop_transform)
        x = torch.ones(1, 4, 224, 224)

        x_out = jit_crop_transform(x)

        self.assertEqual(list(x_out.shape), [1, 4, 160, 160])


class TestTransformationRobustness(BaseTest):
    def test_transform_robustness_init(self) -> None:
        transform_robustness = transforms.TransformationRobustness()
        self.assertIsInstance(
            transform_robustness.padding_transform, torch.nn.ConstantPad2d
        )
        self.assertIsInstance(
            transform_robustness.jitter_transforms, torch.nn.Sequential
        )
        for module in transform_robustness.jitter_transforms:
            self.assertIsInstance(module, transforms.RandomSpatialJitter)
        self.assertIsInstance(transform_robustness.random_scale, transforms.RandomScale)
        # self.assertIsInstance(
        #    transform_robustness.random_rotation, transforms.RandomRotation
        # )
        self.assertIsInstance(
            transform_robustness.final_jitter, transforms.RandomSpatialJitter
        )
        self.assertFalse(transform_robustness.crop_or_pad_output)

    def test_transform_robustness_init_transform_values(self) -> None:
        transform_robustness = transforms.TransformationRobustness()
        self.assertEqual(transform_robustness.padding_transform.padding, (2, 2, 2, 2))
        self.assertEqual(transform_robustness.padding_transform.value, 0.5)

        self.assertEqual(len(transform_robustness.jitter_transforms), 10)
        for module in transform_robustness.jitter_transforms:
            self.assertEqual(module.pad_range, 2 * 4)

        expected_scale = [0.995 ** n for n in range(-5, 80)] + [
            0.998 ** n for n in 2 * list(range(20, 40))
        ]
        self.assertEqual(transform_robustness.random_scale.scale, expected_scale)
        # expected_degrees = (
        #    list(range(-20, 20)) + list(range(-10, 10)) + list(range(-5, 5)) + 5 * [0]
        # )
        # expected_degrees = [float(d) for d in expected_degrees]
        # self.assertEqual(
        #    transform_robustness.random_rotation.degrees, test_expected_degrees
        # )

        self.assertEqual(transform_robustness.final_jitter.pad_range, 2 * 2)

    def test_transform_robustness_init_single_translate(self) -> None:
        transform_robustness = transforms.TransformationRobustness(translate=4)
        self.assertIsInstance(
            transform_robustness.jitter_transforms, transforms.RandomSpatialJitter
        )

    def test_transform_robustness_forward(self) -> None:
        transform_robustness = transforms.TransformationRobustness()
        test_input = torch.ones(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        self.assertTrue(torch.is_tensor(test_output))

    def test_transform_robustness_forward_padding(self) -> None:
        pad_module = torch.nn.ConstantPad2d(2, value=0.5)
        transform_robustness = transforms.TransformationRobustness(
            padding_transform=pad_module
        )
        test_input = torch.ones(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        self.assertTrue(torch.is_tensor(test_output))

    def test_transform_robustness_forward_no_padding(self) -> None:
        transform_robustness = transforms.TransformationRobustness(
            padding_transform=None
        )
        test_input = torch.ones(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        self.assertTrue(torch.is_tensor(test_output))

    def test_transform_robustness_forward_crop_output(self) -> None:
        transform_robustness = transforms.TransformationRobustness(
            padding_transform=None, crop_or_pad_output=True
        )
        test_input = torch.ones(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        self.assertEqual(test_output.shape, test_input.shape)

    def test_transform_robustness_forward_padding_crop_output(self) -> None:
        pad_module = torch.nn.ConstantPad2d(2, value=0.5)
        transform_robustness = transforms.TransformationRobustness(
            padding_transform=pad_module, crop_or_pad_output=True
        )
        test_input = torch.ones(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        self.assertEqual(test_output.shape, test_input.shape)

    def test_transform_robustness_forward_all_disabled(self) -> None:
        transform_robustness = transforms.TransformationRobustness(
            padding_transform=None,
            translate=None,
            scale=None,
            degrees=None,
            final_translate=None,
            crop_or_pad_output=False,
        )
        test_input = torch.randn(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        assertTensorAlmostEqual(self, test_output, test_input, 0)

    def test_transform_robustness_forward_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping TransformationRobustness JIT module test due"
                + " to insufficient Torch version."
            )
        transform_robustness = transforms.TransformationRobustness()
        jit_transform_robustness = torch.jit.script(transform_robustness)
        test_input = torch.ones(1, 3, 224, 224)
        test_output = jit_transform_robustness(test_input)
        self.assertTrue(torch.is_tensor(test_output))

    def test_transform_robustness_forward_padding_crop_output_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping TransformationRobustness with crop or pad output"
                + " JIT module test due to insufficient Torch version."
            )
        pad_module = torch.nn.ConstantPad2d(2, value=0.5)
        transform_robustness = transforms.TransformationRobustness(
            padding_transform=pad_module, crop_or_pad_output=True
        )
        test_input = torch.ones(1, 3, 224, 224)
        test_output = transform_robustness(test_input)
        self.assertEqual(test_output.shape, test_input.shape)
