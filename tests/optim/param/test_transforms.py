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
    def test_random_scale_scale(self) -> None:
        scale_module = transforms.RandomScale(scale=[1, 0.975, 1.025, 0.95, 1.05])
        self.assertEqual(scale_module.scale, [1.0, 0.975, 1.025, 0.95, 1.05])

    def test_random_scale_scale_distributions(self) -> None:
        scale = torch.distributions.Uniform(0.95, 1.05)
        scale_module = transforms.RandomScale(scale=scale)
        self.assertIsInstance(
            scale_module.scale_distribution,
            torch.distributions.distribution.Distribution,
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

    def test_random_scale_matrix(self) -> None:
        scale_module = transforms.RandomScale(scale=[0.5])
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
    def test_random_scale_affine_scale(self) -> None:
        scale_module = transforms.RandomScaleAffine(scale=[1, 0.975, 1.025, 0.95, 1.05])
        self.assertEqual(scale_module.scale, [1.0, 0.975, 1.025, 0.95, 1.05])

    def test_random_scale_affine_scale_distributions(self) -> None:
        scale = torch.distributions.Uniform(0.95, 1.05)
        scale_module = transforms.RandomScaleAffine(scale=scale)
        self.assertIsInstance(
            scale_module.scale_distribution,
            torch.distributions.distribution.Distribution,
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
    def test_center_crop_one_number(self) -> None:
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

    def test_center_crop_two_numbers(self) -> None:
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

    def test_center_crop_one_number_exact(self) -> None:
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

    def test_center_crop_two_numbers_exact(self) -> None:
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

    def test_center_crop_offset_left_uneven_sides(self) -> None:
        crop_mod = transforms.CenterCrop(
            [5, 5], pixels_from_edges=False, offset_left=True
        )
        x = torch.ones(1, 3, 5, 5)
        px = F.pad(x, (5, 4, 5, 4), value=float("-inf"))
        cropped_tensor = crop_mod(px)
        assertTensorAlmostEqual(self, x, cropped_tensor)

    def test_center_crop_offset_left_even_sides(self) -> None:
        crop_mod = transforms.CenterCrop(
            [5, 5], pixels_from_edges=False, offset_left=True
        )
        x = torch.ones(1, 3, 5, 5)
        px = F.pad(x, (5, 5, 5, 5), value=float("-inf"))
        cropped_tensor = crop_mod(px)
        assertTensorAlmostEqual(self, x, cropped_tensor)

    def test_center_crop_one_number_exact_jit_module(self) -> None:
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


class TestBlendAlpha(BaseTest):
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
    def test_to_rgb_i1i2i3(self) -> None:
        to_rgb = transforms.ToRGB(transform="i1i2i3")
        to_rgb_np = numpy_transforms.ToRGB(transform="i1i2i3")
        assertArraysAlmostEqual(to_rgb.transform.numpy(), to_rgb_np.transform)

    def test_to_rgb_klt(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        to_rgb_np = numpy_transforms.ToRGB(transform="klt")
        assertArraysAlmostEqual(to_rgb.transform.numpy(), to_rgb_np.transform)

    def test_to_rgb_custom(self) -> None:
        matrix = torch.eye(3, 3)
        to_rgb = transforms.ToRGB(transform=matrix)
        to_rgb_np = numpy_transforms.ToRGB(transform=matrix.numpy())
        assertArraysAlmostEqual(to_rgb.transform.numpy(), to_rgb_np.transform)

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
