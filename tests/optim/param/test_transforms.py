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


class TestRandSelect(BaseTest):
    def test_rand_select(self) -> None:
        a = (1, 2, 3, 4, 5)
        b = torch.Tensor([0.1, -5, 56.7, 99.0])

        self.assertIn(transforms._rand_select(a), a)
        self.assertIn(transforms._rand_select(b), b)


class TestRandomScale(BaseTest):
    def test_random_scale(self) -> None:
        scale_module = transforms.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)

        # Test rescaling
        assertTensorAlmostEqual(
            self,
            scale_module.scale_tensor(test_tensor, 0.5),
            torch.ones(3, 1).repeat(3, 1, 3).unsqueeze(0),
            0,
        )

        assertTensorAlmostEqual(
            self,
            scale_module.scale_tensor(test_tensor, 1.5),
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

    def test_random_scale_matrix(self) -> None:
        scale_module = transforms.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)
        # Test scale matrices

        assertTensorAlmostEqual(
            self,
            scale_module.get_scale_mat(0.5, test_tensor.device, test_tensor.dtype),
            torch.tensor([[0.5000, 0.0000, 0.0000], [0.0000, 0.5000, 0.0000]]),
            0,
        )

        assertTensorAlmostEqual(
            self,
            scale_module.get_scale_mat(1.24, test_tensor.device, test_tensor.dtype),
            torch.tensor([[1.2400, 0.0000, 0.0000], [0.0000, 1.2400, 0.0000]]),
            0,
        )


class TestRandomRotation(BaseTest):
    def test_random_rotation_init(self) -> None:
        test_degrees = [0.0, 1.0, 2.0, 3.0, 4.0]
        rotation_module = transforms.RandomRotation(test_degrees)
        degrees = rotation_module.degrees
        self.assertTrue(hasattr(degrees, "__iter__"))
        self.assertEqual(degrees, test_degrees)
        self.assertFalse(rotation_module._is_distribution)
        self.assertEqual(rotation_module.mode, "bilinear")
        self.assertEqual(rotation_module.padding_mode, "zeros")
        self.assertFalse(rotation_module.align_corners)

    def test_random_rotation_tensor_degrees(self) -> None:
        degrees = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        rotation_module = transforms.RandomRotation(degrees=degrees)
        self.assertEqual(rotation_module.degrees, degrees.tolist())

    def test_random_rotation_int_degrees(self) -> None:
        degrees = [1, 2, 3, 4, 5]
        rotation_module = transforms.RandomRotation(degrees=degrees)
        for r in rotation_module.degrees:
            self.assertIsInstance(r, float)
        self.assertEqual(rotation_module.degrees, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_random_rotation_degrees_distributions(self) -> None:
        degrees = torch.distributions.Uniform(0.95, 1.05)
        rotation_module = transforms.RandomRotation(degrees=degrees)
        self.assertIsInstance(
            rotation_module.degrees_distribution,
            torch.distributions.distribution.Distribution,
        )
        self.assertTrue(rotation_module._is_distribution)

    def test_random_rotation_torch_version_check(self) -> None:
        rotation_module = transforms.RandomRotation([1.0])
        _has_align_corners = torch.__version__ >= "1.3.0"
        self.assertEqual(rotation_module._has_align_corners, _has_align_corners)

    def test_random_rotation_matrix(self) -> None:
        theta = 25.1
        rotation_module = transforms.RandomRotation([theta])
        rot_matrix = rotation_module._get_rot_mat(
            theta, device=torch.device("cpu"), dtype=torch.float32
        )
        expected_matrix = torch.tensor(
            [[0.9056, -0.4242, 0.0000], [0.4242, 0.9056, 0.0000]]
        )

        assertTensorAlmostEqual(self, rot_matrix, expected_matrix)

    def test_random_rotation_rotate_tensor(self) -> None:
        rotation_module = transforms.RandomRotation([25.0])

        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        test_output = rotation_module._rotate_tensor(test_input, 25.0)

        expected_output = (
            torch.tensor(
                [
                    [0.1143, 0.0000, 0.0000, 0.0000],
                    [0.5258, 0.6198, 0.2157, 0.0000],
                    [0.0000, 0.2157, 0.6198, 0.5258],
                    [0.0000, 0.0000, 0.0000, 0.1143],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        assertTensorAlmostEqual(self, test_output, expected_output, 0.005)

    def test_random_rotation_forward_exact(self) -> None:
        rotation_module = transforms.RandomRotation([25.0])

        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        test_output = rotation_module(test_input)

        expected_output = (
            torch.tensor(
                [
                    [0.1143, 0.0000, 0.0000, 0.0000],
                    [0.5258, 0.6198, 0.2157, 0.0000],
                    [0.0000, 0.2157, 0.6198, 0.5258],
                    [0.0000, 0.0000, 0.0000, 0.1143],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        assertTensorAlmostEqual(self, test_output, expected_output, 0.005)

    def test_random_rotation_forward_exact_nearest_reflection(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping RandomRotation forward due exact nearest relfection"
                + " to insufficient Torch version."
            )
        rotation_module = transforms.RandomRotation(
            [45.0], mode="nearest", padding_mode="reflection"
        )
        self.assertEqual(rotation_module.mode, "nearest")
        self.assertEqual(rotation_module.padding_mode, "reflection")

        test_input = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()
        test_output = rotation_module(test_input)

        expected_output = torch.tensor(
            [
                [
                    [
                        [2.0, 2.0, 7.0, 11.0],
                        [1.0, 6.0, 10.0, 11.0],
                        [4.0, 9.0, 10.0, 14.0],
                        [8.0, 8.0, 13.0, 14.0],
                    ]
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, 0.0)

    def test_random_rotation_forward_exact_nearest(self) -> None:
        rotation_module = transforms.RandomRotation([45.0], mode="nearest")
        self.assertEqual(rotation_module.mode, "nearest")
        self.assertEqual(rotation_module.padding_mode, "zeros")

        test_input = torch.arange(0, 1 * 1 * 4 * 4).view(1, 1, 4, 4).float()
        test_output = rotation_module(test_input)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 2.0, 7.0, 0.0],
                        [1.0, 6.0, 10.0, 11.0],
                        [4.0, 9.0, 10.0, 14.0],
                        [0.0, 8.0, 13.0, 0.0],
                    ]
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, 0.0)

    def test_random_rotation_forward(self) -> None:
        degrees = list(range(-25, 25))
        rotation_module = transforms.RandomRotation(degrees=degrees)
        test_tensor = torch.ones(1, 3, 10, 10)
        output_tensor = rotation_module(test_tensor)
        self.assertEqual(list(output_tensor.shape), list(test_tensor.shape))

    def test_random_rotation_forward_distributions(self) -> None:
        degrees = torch.distributions.Uniform(-25, 25)
        rotation_module = transforms.RandomRotation(degrees=degrees)
        test_tensor = torch.ones(1, 3, 10, 10)
        output_tensor = rotation_module(test_tensor)
        self.assertEqual(list(output_tensor.shape), list(test_tensor.shape))

    def test_random_rotation_forward_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping RandomRotation forward CUDA test due to not supporting"
                + " CUDA."
            )
        rotate_transform = transforms.RandomRotation(list(range(-25, 25)))
        x = torch.ones(1, 3, 224, 224).cuda()
        output = rotate_transform(x)

        self.assertTrue(output.is_cuda)
        self.assertEqual(output.shape, x.shape)

    def test_random_rotation_matrix_torch_math_module(self) -> None:
        theta = 25.1
        rotation_module = transforms.RandomRotation([theta])
        rot_matrix = rotation_module._get_rot_mat(
            theta, device=torch.device("cpu"), dtype=torch.float32
        )

        theta_expected = torch.tensor(theta) * 3.141592653589793 / 180.0
        expected_matrix = torch.tensor(
            [
                [torch.cos(theta_expected), -torch.sin(theta_expected), 0.0],
                [torch.sin(theta_expected), torch.cos(theta_expected), 0.0],
            ],
        )

        assertTensorAlmostEqual(self, rot_matrix, expected_matrix, 0.0)

    def test_random_rotation_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping RandomRotation JIT test due to insufficient Torch version."
            )
        rotation_module = transforms.RandomRotation([25.0])
        jit_rotation_module = torch.jit.script(rotation_module)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)

        test_output = jit_rotation_module(test_input)
        expected_output = (
            torch.tensor(
                [
                    [0.1143, 0.0000, 0.0000, 0.0000],
                    [0.5258, 0.6198, 0.2157, 0.0000],
                    [0.0000, 0.2157, 0.6198, 0.5258],
                    [0.0000, 0.0000, 0.0000, 0.1143],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        assertTensorAlmostEqual(self, test_output, expected_output, 0.005)


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


class TestIgnoreAlpha(BaseTest):
    def test_ignore_alpha(self) -> None:
        ignore_alpha = transforms.IgnoreAlpha()
        test_input = torch.ones(1, 4, 3, 3)
        rgb_tensor = ignore_alpha(test_input)
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


class TestScaleInputRange(BaseTest):
    def test_scale_input_range(self) -> None:
        x = torch.ones(1, 3, 4, 4)
        scale_input = transforms.ScaleInputRange(255)
        output_tensor = scale_input(x)
        self.assertEqual(output_tensor.mean(), 255.0)


class TestRGBToBGR(BaseTest):
    def test_rgb_to_bgr(self) -> None:
        x = torch.randn(1, 3, 224, 224)
        rgb_to_bgr = transforms.RGBToBGR()
        output_tensor = rgb_to_bgr(x)
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
