#!/usr/bin/env python3

from typing import List, Tuple, cast

import torch

from captum._utils.common import _select_targets

from ..helpers.basic import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
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
