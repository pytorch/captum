#!/usr/bin/env python3

import torch

from captum.attr._utils.common import (
    _validate_input,
    _validate_noise_tunnel_type,
    _select_targets,
)
from captum.attr._utils.common import MaxList
from captum.attr._core.noise_tunnel import SUPPORTED_NOISE_TUNNEL_TYPES

from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_validate_input(self):
        with self.assertRaises(AssertionError):
            _validate_input(torch.Tensor([-1.0, 1.0]), torch.Tensor([-2.0]))
            _validate_input(
                torch.Tensor([-1.0, 1.0]), torch.Tensor([-1.0, 1.0]), n_steps=-1
            )
            _validate_input(
                torch.Tensor([-1.0, 1.0]), torch.Tensor([-1.0, 1.0]), method="abcde"
            )
        _validate_input(torch.Tensor([-1.0]), torch.Tensor([-2.0]))
        _validate_input(
            torch.Tensor([-1.0]), torch.Tensor([-2.0]), method="gausslegendre"
        )

    def test_validate_nt_type(self):
        with self.assertRaises(AssertionError):
            _validate_noise_tunnel_type("abc", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad_sq", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("vargrad", SUPPORTED_NOISE_TUNNEL_TYPES)

    def test_select_target_2d(self):
        output_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assertTensorAlmostEqual(self, _select_targets(output_tensor, 1), [2, 5, 8])
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, torch.tensor(0)), [1, 4, 7]
        )
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, torch.tensor([1, 2, 0])), [2, 6, 7]
        )
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, [1, 2, 0]), [2, 6, 7]
        )

        # Verify error is raised if too many dimensions are provided.
        with self.assertRaises(AssertionError):
            _select_targets(output_tensor, (1, 2))

    def test_select_target_3d(self):
        output_tensor = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
        )
        assertTensorAlmostEqual(self, _select_targets(output_tensor, (0, 1)), [2, 8])
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, [(0, 1), (2, 0)]), [2, 3]
        )

        # Verify error is raised if list is longer than number of examples.
        with self.assertRaises(AssertionError):
            _select_targets(output_tensor, [(0, 1), (2, 0), (3, 2)])

        # Verify error is raised if too many dimensions are provided.
        with self.assertRaises(AssertionError):
            _select_targets(output_tensor, (1, 2, 3))

    def test_max_list(self):
        ml = MaxList(3)
        ml.add(5)
        ml.add(2)
        ml.add(1)

        self.assertEqual(ml.get_list(), [5, 2, 1])

        ml.add(3)
        ml.add(1)

        self.assertEqual(ml.get_list(), [5, 3, 2])

    def test_max_item(self):
        ml = MaxList(1)
        ml.add(5)
        ml.add(2)
        ml.add(8)
        ml.add(3)

        self.assertEqual(ml.get_list(), [8])

    def test_max_list_string(self):
        ml = MaxList(4, key=lambda x: len(x))
        ml.add("American Trade Deal")
        ml.add("Chryseler")
        ml.add("Supercalafragilisticexpialodocious")
        ml.add("ravenclaw")
        ml.add("Facebook Rocks!")

        self.assertEqual(ml.get_list()[3], "Facebook Rocks!")
        self.assertEqual(len(ml.get_list()), 4)
