#!/usr/bin/env python3

from typing import List, Tuple, cast

import torch

from captum.attr._core.noise_tunnel import SUPPORTED_NOISE_TUNNEL_TYPES
from captum.attr._utils.common import (
    _select_targets,
    _validate_input,
    _validate_noise_tunnel_type,
)

from .helpers.utils import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_validate_input(self) -> None:
        with self.assertRaises(AssertionError):
            _validate_input((torch.tensor([-1.0, 1.0]),), (torch.tensor([-2.0]),))
            _validate_input(
                (torch.tensor([-1.0, 1.0]),), (torch.tensor([-1.0, 1.0]),), n_steps=-1
            )
            _validate_input(
                (torch.tensor([-1.0, 1.0]),),
                (torch.tensor([-1.0, 1.0]),),
                method="abcde",
            )
        _validate_input((torch.tensor([-1.0]),), (torch.tensor([-2.0]),))
        _validate_input(
            (torch.tensor([-1.0]),), (torch.tensor([-2.0]),), method="gausslegendre"
        )

    def test_validate_nt_type(self) -> None:
        with self.assertRaises(AssertionError):
            _validate_noise_tunnel_type("abc", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad_sq", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("vargrad", SUPPORTED_NOISE_TUNNEL_TYPES)

    def test_select_target_2d(self) -> None:
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

    def test_select_target_3d(self) -> None:
        output_tensor = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
        )
        assertTensorAlmostEqual(self, _select_targets(output_tensor, (0, 1)), [2, 8])
        assertTensorAlmostEqual(
            self,
            _select_targets(
                output_tensor, cast(List[Tuple[int, ...]], [(0, 1), (2, 0)])
            ),
            [2, 3],
        )

        # Verify error is raised if list is longer than number of examples.
        with self.assertRaises(AssertionError):
            _select_targets(
                output_tensor, cast(List[Tuple[int, ...]], [(0, 1), (2, 0), (3, 2)])
            )

        # Verify error is raised if too many dimensions are provided.
        with self.assertRaises(AssertionError):
            _select_targets(output_tensor, (1, 2, 3))
