#!/usr/bin/env python3
import unittest
from os import path
from typing import List

import captum.optim._param.image.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.optim.helpers import numpy_transforms

try:
    from torchtext.transforms import CLIPTokenizer as CLIPTokenizer_TorchText

    _torchtext_has_clip_tokenizer = True
except ImportError:
    _torchtext_has_clip_tokenizer = False


class TestRandomScale(BaseTest):
    def test_random_scale_init(self) -> None:
        scale_module = transforms.RandomScale(scale=[1, 0.975, 1.025, 0.95, 1.05])
        self.assertEqual(scale_module.scale, [1.0, 0.975, 1.025, 0.95, 1.05])
        self.assertFalse(scale_module._is_distribution)
        self.assertEqual(scale_module.mode, "bilinear")
        self.assertFalse(scale_module.align_corners)
        self.assertFalse(scale_module.recompute_scale_factor)
        self.assertFalse(scale_module.antialias)

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

        has_antialias = version.parse(torch.__version__) >= version.parse("1.11.0")
        self.assertEqual(scale_module._has_antialias, has_antialias)

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

    def test_random_scale_antialias(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.11.0"):
            raise unittest.SkipTest(
                "Skipping RandomScale antialias test"
                + " due to insufficient Torch version."
            )
        scale_module = transforms.RandomScale(scale=[0.5], antialias=True)
        test_tensor = torch.arange(0, 1 * 1 * 10 * 10).view(1, 1, 10, 10).float()

        scaled_tensor = scale_module._scale_tensor(test_tensor, 0.5)

        expected_tensor = torch.tensor(
            [
                [
                    [
                        [7.8571, 9.6429, 11.6429, 13.6429, 15.4286],
                        [25.7143, 27.5000, 29.5000, 31.5000, 33.2857],
                        [45.7143, 47.5000, 49.5000, 51.5000, 53.2857],
                        [65.7143, 67.5000, 69.5000, 71.5000, 73.2857],
                        [83.5714, 85.3571, 87.3571, 89.3571, 91.1429],
                    ]
                ]
            ]
        )

        assertTensorAlmostEqual(
            self,
            scaled_tensor,
            expected_tensor,
            0.0005,
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        jittered_array = torch.as_tensor(jittered_array)

        assertTensorAlmostEqual(self, jittered_tensor[0], jittered_array, 0, mode="max")
        assertTensorAlmostEqual(self, jittered_tensor[1], jittered_array, 0, mode="max")
        assertTensorAlmostEqual(self, jittered_tensor[2], jittered_array, 0, mode="max")

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
        jittered_array = torch.as_tensor(jittered_array)

        assertTensorAlmostEqual(self, jittered_tensor[0], jittered_array, 0, mode="max")
        assertTensorAlmostEqual(self, jittered_tensor[1], jittered_array, 0, mode="max")
        assertTensorAlmostEqual(self, jittered_tensor[2], jittered_array, 0, mode="max")

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
        jittered_array = torch.as_tensor(jittered_array)

        assertTensorAlmostEqual(self, jittered_tensor[0], jittered_array, 0, mode="max")
        assertTensorAlmostEqual(self, jittered_tensor[1], jittered_array, 0, mode="max")
        assertTensorAlmostEqual(self, jittered_tensor[2], jittered_array, 0, mode="max")

    def test_random_spatial_jitter_forward(self) -> None:
        t_val = 3

        spatialjitter = transforms.RandomSpatialJitter(t_val)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        jittered_tensor = spatialjitter(test_input)
        self.assertEqual(list(jittered_tensor.shape), list(test_input.shape))

    def test_random_spatial_jitter_forward_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")

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
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor[None, None, :], 0)

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
        if version.parse(torch.__version__) <= version.parse("1.9.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
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
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor[None, None, :], 0)


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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        cropped_array = torch.as_tensor(cropped_array)

        assertTensorAlmostEqual(self, cropped_tensor, cropped_array, 0, mode="max")
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
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor[None, None, :], 0)

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
        if version.parse(torch.__version__) <= version.parse("1.9.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
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
        assertTensorAlmostEqual(self, cropped_tensor, expected_tensor[None, None, :], 0)


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
        blended_array = torch.as_tensor(blended_array)

        assertTensorAlmostEqual(self, blended_tensor, blended_array, 0, mode="max")

    def test_blend_alpha_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        blended_array = torch.as_tensor(blended_array)

        assertTensorAlmostEqual(self, blended_tensor, blended_array, 0, mode="max")


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
        assertTensorAlmostEqual(
            self, to_rgb.transform, torch.as_tensor(to_rgb_np.transform), mode="max"
        )
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

        assertTensorAlmostEqual(
            self, to_rgb.transform, torch.as_tensor(to_rgb_np.transform), mode="max"
        )
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
        assertTensorAlmostEqual(
            self, to_rgb.transform, torch.as_tensor(to_rgb_np.transform), mode="max"
        )
        assertTensorAlmostEqual(self, to_rgb.transform, matrix, 0.0)

    def test_to_rgb_init_value_error(self) -> None:
        with self.assertRaises(ValueError):
            transforms.ToRGB(transform="error")

    def test_to_rgb_klt_forward(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(1, 3, 4, 4).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        expected_rgb_tensor = torch.stack([r, g, b]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)
        self.assertEqual(list(rgb_tensor.names), ["B", "C", "H", "W"])

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_alpha_klt_forward(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(1, 4, 4, 4).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        a = torch.ones(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b, a]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)
        self.assertEqual(list(rgb_tensor.names), ["B", "C", "H", "W"])

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_alpha_klt_forward_dim_3(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(4, 4, 4).refine_names("C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        a = torch.ones(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b, a])

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)
        self.assertEqual(list(rgb_tensor.names), ["C", "H", "W"])

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_klt_forward_no_named_dims(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(1, 3, 4, 4)
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        expected_rgb_tensor = torch.stack([r, g, b]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)
        self.assertEqual(list(rgb_tensor.names), [None] * 4)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_alpha_klt_forward_no_named_dims(self) -> None:
        to_rgb = transforms.ToRGB(transform="klt")
        test_tensor = torch.ones(1, 4, 4, 4)
        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4, 4) * 0.8009
        g = torch.ones(4, 4) * 0.4762
        b = torch.ones(4, 4) * 0.4546
        a = torch.ones(4, 4)
        expected_rgb_tensor = torch.stack([r, g, b, a]).unsqueeze(0)

        assertTensorAlmostEqual(self, rgb_tensor, expected_rgb_tensor, 0.002)
        self.assertEqual(list(rgb_tensor.names), [None] * 4)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_i1i2i3_forward(self) -> None:
        to_rgb = transforms.ToRGB(transform="i1i2i3")
        test_tensor = torch.ones(1, 3, 4, 4).refine_names("B", "C", "H", "W")
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
        to_rgb = transforms.ToRGB(transform="i1i2i3")
        test_tensor = torch.ones(1, 4, 4, 4).refine_names("B", "C", "H", "W")
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
        matrix = torch.eye(3, 3)
        to_rgb = transforms.ToRGB(transform=matrix)
        test_tensor = torch.ones(1, 3, 4, 4).refine_names("B", "C", "H", "W")
        rgb_tensor = to_rgb(test_tensor)

        to_rgb_np = numpy_transforms.ToRGB(transform=matrix.numpy())
        test_array = np.ones((1, 3, 4, 4))
        rgb_array = to_rgb_np.to_rgb(test_array)

        assertTensorAlmostEqual(
            self, rgb_tensor, torch.as_tensor(rgb_array), mode="max"
        )

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)
        assertTensorAlmostEqual(
            self, inverse_tensor, torch.ones_like(inverse_tensor.rename(None))
        )

    def test_to_rgb_klt_forward_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
            raise unittest.SkipTest(
                "Skipping ToRGB forward JIT module test due to insufficient"
                + " Torch version."
            )
        to_rgb = transforms.ToRGB(transform="klt")
        jit_to_rgb = torch.jit.script(to_rgb)
        test_tensor = torch.ones(1, 3, 4, 4)
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        self.assertIsInstance(
            transform_robustness.random_rotation, transforms.RandomRotation
        )
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

        expected_scale = [0.995**n for n in range(-5, 80)] + [
            0.998**n for n in 2 * list(range(20, 40))
        ]
        self.assertEqual(transform_robustness.random_scale.scale, expected_scale)
        expected_degrees = (
            list(range(-20, 20)) + list(range(-10, 10)) + list(range(-5, 5)) + 5 * [0]
        )
        expected_degrees = [float(d) for d in expected_degrees]
        self.assertEqual(transform_robustness.random_rotation.degrees, expected_degrees)

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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
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


class TestCLIPTokenizer(BaseTest):
    def test_clip_tokenizer_pretrained_download(self) -> None:
        file_path = path.join(
            torch.hub.get_dir(), "vocab", "clip_bpe_simple_vocab_48895.txt"
        )
        merges_path = transforms.CLIPTokenizer._download_clip_bpe_merges(None)
        self.assertEqual(merges_path, file_path)

    def test_clip_tokenizer_pretrained_download_custom_path(self) -> None:
        custom_path = path.join(torch.hub.get_dir(), "vocab_test")
        file_path = path.join(custom_path, "clip_bpe_simple_vocab_48895.txt")
        merges_path = transforms.CLIPTokenizer._download_clip_bpe_merges(custom_path)
        self.assertEqual(merges_path, file_path)

    def test_clip_tokenizer_pretrained_download_assert_error(self) -> None:
        file_path = path.join("vocab", "clip_bpe_simple_vocab_48895.txt")
        with self.assertRaises(AssertionError):
            _ = transforms.CLIPTokenizer._download_clip_bpe_merges(file_path)

    def test_clip_tokenizer_init(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer init test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)

        self.assertEqual(clip_tokenizer.context_length, 77)
        self.assertEqual(clip_tokenizer.start_token, "<|startoftext|>")
        self.assertEqual(clip_tokenizer.end_token, "<|endoftext|>")
        self.assertIsNone(clip_tokenizer._num_merges)
        self.assertEqual(clip_tokenizer.padding_value, 0)
        self.assertFalse(clip_tokenizer.truncate)

        file_path = path.join(
            torch.hub.get_dir(), "vocab", "clip_bpe_simple_vocab_48895.txt"
        )
        self.assertEqual(clip_tokenizer._merges_path, file_path)
        self.assertIsInstance(
            clip_tokenizer.clip_tokenizer_module, CLIPTokenizer_TorchText
        )

    def test_clip_tokenizer_str_input(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_str_input_context_length_54(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input"
                + " context_length test"
            )
        context_length = 54
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, context_length=context_length
        )
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_str_input_context_length_padding(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input test"
            )
        padding_value = -1
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, padding_value=padding_value
        )
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [padding_value] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_list_str_input(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer list str input"
                + " test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = ["this is a test!", "a picture of a cat."]

        text_output = clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [
            [49406, 589, 533, 320, 1628, 256, 49407],
            [49406, 320, 1674, 539, 320, 2368, 269, 49407],
        ]

        self.assertEqual(list(text_output.shape), [2, context_length])
        for b, t in enumerate(token_ids):
            padding = [0] * (context_length - len(t))
            token_set = t + padding
            self.assertEqual(text_output[b].tolist(), token_set)

    def test_clip_tokenizer_str_input_decode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input"
                + " decode test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "This is a test!"

        text_output = clip_tokenizer(text_input_str)
        text_output_str = clip_tokenizer.decode(text_output)

        expected_ouput_str = ["this is a test !"]
        self.assertEqual(text_output_str, expected_ouput_str)

    def test_clip_tokenizer_str_input_decode_special_tokens(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input"
                + " decode include_special_tokens test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "This is a test!"

        text_output = clip_tokenizer(text_input_str)
        text_output_str = clip_tokenizer.decode(
            text_output, include_special_tokens=True
        )

        expected_ouput_str = ["<|startoftext|>this is a test ! <|endoftext|>"]
        self.assertEqual(text_output_str, expected_ouput_str)

    def test_clip_tokenizer_list_decode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer list decode"
                + " test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)

        token_list = [49406, 589, 533, 320, 1628, 256, 49407, 0]

        str_output = clip_tokenizer.decode(token_list)
        expected_str = ["this is a test !"]
        self.assertEqual(str_output, expected_str)

    def test_clip_tokenizer_list_of_list_decode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer list of list"
                + " decode test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)

        token_list = [
            [49406, 589, 533, 320, 1628, 256, 49407],
            [49406, 320, 1674, 539, 320, 2368, 269, 49407, 0, 0],
        ]

        str_output = clip_tokenizer.decode(token_list)
        expected_str = ["this is a test !", "a picture of a cat ."]
        self.assertEqual(str_output, expected_str)

    def test_clip_tokenizer_no_special_tokens(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer no special"
                + " tokens test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, start_token=None, end_token=None
        )
        text_input_str = "This is a test!"

        text_output = clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [589, 533, 320, 1628, 256]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

        text_output_str = clip_tokenizer.decode(
            text_output, include_special_tokens=True
        )

        expected_ouput_str = ["this is a test !"]
        self.assertEqual(text_output_str, expected_ouput_str)

    def test_clip_tokenizer_pretrained_merges_false(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer pretrained"
                + " merges False test"
            )
        merges_path = transforms.CLIPTokenizer._download_clip_bpe_merges(None)
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=False, merges_path=merges_path
        )
        text_input_str = "This is a test!"

        text_output = clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

        text_output_str = clip_tokenizer.decode(text_output)

        expected_ouput_str = ["this is a test !"]
        self.assertEqual(text_output_str, expected_ouput_str)

    def test_clip_tokenizer_str_input_jit(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input JIT"
                + " test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        jit_clip_tokenizer = torch.jit.script(clip_tokenizer)
        text_output = jit_clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_unicode_encode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer unicode test"
            )

        from torchtext import __version__ as torchtext_version

        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, context_length=376
        )

        bpe_v = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        bpe_vocab = [chr(c) for c in bpe_v + [256 + n for n in list(range(0, 68))]]
        bpe_vocab_str = " ".join(bpe_vocab)
        txt_output = clip_tokenizer(bpe_vocab_str)

        # fmt: off
        expected_tokens = [
            256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
            333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 314, 315,
            316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
            331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
            346, 347, 348, 349, 10830, 41359, 1950, 126, 353, 20199, 126, 355, 126,
            356, 126, 357, 5811, 126, 359, 14434, 126, 361, 8436, 43593, 6858, 126,
            365, 41175, 126, 367, 12559, 126, 369, 126, 370, 14844, 126, 372, 126,
            373, 28059, 7599, 126, 376, 33613, 126, 378, 17133, 21259, 22229, 127,
            351, 127, 352, 47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359,
            40520, 127, 361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366,
            27733, 127, 368, 127, 369, 37423, 16165, 45598, 127, 373, 36019, 127, 375,
            47177, 127, 377, 127, 378, 127, 509, 21259, 22229, 127, 351, 127, 352,
            47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359, 40520, 127,
            361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366, 27733, 127,
            368, 127, 369, 37423, 127, 371, 45598, 127, 373, 36019, 127, 375, 47177,
            127, 377, 127, 378, 127, 379, 128, 479, 128, 479, 128, 481, 128, 481, 128,
            483, 128, 483, 31719, 31719, 128, 487, 128, 487, 128, 489, 128, 489, 128,
            491, 128, 491, 128, 493, 128, 493, 128, 495, 128, 495, 128, 497, 128, 497,
            128, 499, 128, 499, 128, 501, 128, 501, 128, 503, 128, 503, 128, 505, 128,
            505, 128, 507, 128, 507, 128, 509, 128, 509, 128, 350, 128, 350, 128, 352,
            128, 352, 128, 354, 128, 354, 128, 356, 128, 356, 128, 358, 128, 358, 128,
            360, 128, 360, 128, 511, 128, 511, 128, 363, 128, 363, 328, 16384, 41901,
            72, 329, 72, 329, 128, 369, 128, 369, 128, 371, 128, 371, 128, 372, 128,
            374, 128, 374, 128, 376, 128, 376, 128, 378, 128, 378, 129, 478, 129, 478,
            129, 480, 129, 480, 129, 482,
        ]
        # fmt: on

        if torchtext_version <= "0.12.0":
            # Correct for TorchText bug
            expected_tokens[338:342] = [128, 367, 128, 367]

        self.assertEqual(txt_output[0].tolist()[1:-1], expected_tokens)

    def test_clip_tokenizer_unicode_decode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer unicode decode"
                + " test"
            )

        # fmt: off
        input_tokens = [
            256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
            333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 314, 315,
            316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
            331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
            346, 347, 348, 349, 10830, 41359, 1950, 126, 353, 20199, 126, 355, 126,
            356, 126, 357, 5811, 126, 359, 14434, 126, 361, 8436, 43593, 6858, 126,
            365, 41175, 126, 367, 12559, 126, 369, 126, 370, 14844, 126, 372, 126,
            373, 28059, 7599, 126, 376, 33613, 126, 378, 17133, 21259, 22229, 127,
            351, 127, 352, 47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359,
            40520, 127, 361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366,
            27733, 127, 368, 127, 369, 37423, 16165, 45598, 127, 373, 36019, 127, 375,
            47177, 127, 377, 127, 378, 127, 509, 21259, 22229, 127, 351, 127, 352,
            47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359, 40520, 127,
            361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366, 27733, 127,
            368, 127, 369, 37423, 127, 371, 45598, 127, 373, 36019, 127, 375, 47177,
            127, 377, 127, 378, 127, 379, 128, 479, 128, 479, 128, 481, 128, 481, 128,
            483, 128, 483, 31719, 31719, 128, 487, 128, 487, 128, 489, 128, 489, 128,
            491, 128, 491, 128, 493, 128, 493, 128, 495, 128, 495, 128, 497, 128, 497,
            128, 499, 128, 499, 128, 501, 128, 501, 128, 503, 128, 503, 128, 505, 128,
            505, 128, 507, 128, 507, 128, 509, 128, 509, 128, 350, 128, 350, 128, 352,
            128, 352, 128, 354, 128, 354, 128, 356, 128, 356, 128, 358, 128, 358, 128,
            360, 128, 360, 128, 511, 128, 511, 128, 363, 128, 363, 328, 16384, 41901,
            72, 329, 72, 329, 128, 369, 128, 369, 128, 371, 128, 371, 128, 372, 128,
            374, 128, 374, 128, 376, 128, 376, 128, 378, 128, 378, 129, 478, 129, 478,
            129, 480, 129, 480, 129, 482,
        ]
        # fmt: on

        input_tokens = torch.as_tensor([input_tokens])
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        txt_output_str = clip_tokenizer.decode(input_tokens)

        expected_str = (
            """!"#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\]^_`abcd"""  # noqa: W605,E501
            + """efghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿àáâãäåæçèé"""
            + """êëìíîïðñòóôõö×øùúûüýþßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿāāăăąąććĉĉċċ"""
            + """ččďďđđēēĕĕėėęęěěĝĝğğġġģģĥĥħħĩĩīīĭĭįįi̇ıijijĵĵķķĸĺĺļļľľŀŀłłń"""
        )
        self.assertEqual(len(txt_output_str), 1)
        self.assertEqual(txt_output_str[0].replace(" ", ""), expected_str)

    def test_clip_tokenizer_truncate(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer truncate test"
            )
        context_length = 5
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, context_length=context_length, truncate=True
        )
        text_input_str = "this is a test!"
        text_output = clip_tokenizer(text_input_str)

        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), [49406, 589, 533, 320, 49407])

    def test_clip_tokenizer_truncate_no_end_token(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer truncate no"
                + " end token test"
            )
        context_length = 5
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True,
            context_length=context_length,
            end_token=None,
            truncate=True,
        )
        text_input_str = "this is a test!"
        text_output = clip_tokenizer(text_input_str)

        self.assertEqual(list(text_output.shape), [1, context_length])
        self.assertEqual(text_output[0].tolist(), [49406, 589, 533, 320, 1628])
