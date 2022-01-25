#!/usr/bin/env python3

from typing import List, Tuple, cast

import torch
from captum._utils.common import safe_div, _reduce_list, _select_targets, _sort_key_list
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_safe_div_number_denom(self):
        num = torch.tensor(4.0)
        assert safe_div(num, 2) == 2.0
        assert safe_div(num, 0, 2) == 2.0
        assert safe_div(num, 2.0) == 2.0
        assert safe_div(num, 0.0, 2.0) == 2.0

    def test_safe_div_tensor_denom(self):
        num = torch.tensor([4.0, 6.0])

        exp = torch.tensor([2.0, 3.0])
        assert (safe_div(num, torch.tensor([2.0, 2.0])) == exp).all()

        # tensor default denom
        assert (safe_div(num, torch.tensor([0.0, 0.0]), torch.tensor(2.0)) == exp).all()
        assert (
            safe_div(
                num,
                torch.tensor([0.0, 0.0]),
                torch.tensor([2.0, 2.0]),
            )
            == exp
        ).all()

        # float default denom
        assert (safe_div(num, torch.tensor([0.0, 0.0]), 2.0) == exp).all()

    def test_reduce_list_tensors(self):
        tensors = [torch.tensor([[3, 4, 5]]), torch.tensor([[0, 1, 2]])]
        reduced = _reduce_list(tensors)
        assertTensorAlmostEqual(self, reduced, [[3, 4, 5], [0, 1, 2]])

    def test_reduce_list_tuples(self):
        tensors = [
            (torch.tensor([[3, 4, 5]]), torch.tensor([[0, 1, 2]])),
            (torch.tensor([[3, 4, 5]]), torch.tensor([[0, 1, 2]])),
        ]
        reduced = _reduce_list(tensors)
        assertTensorAlmostEqual(self, reduced[0], [[3, 4, 5], [3, 4, 5]])
        assertTensorAlmostEqual(self, reduced[1], [[0, 1, 2], [0, 1, 2]])

    def test_sort_key_list(self):
        key_list = [
            torch.device("cuda:13"),
            torch.device("cuda:17"),
            torch.device("cuda:10"),
            torch.device("cuda:0"),
        ]
        device_index_list = [0, 10, 13, 17]
        sorted_keys = _sort_key_list(key_list, device_index_list)
        for i in range(len(key_list)):
            self.assertEqual(sorted_keys[i].index, device_index_list[i])

    def test_sort_key_list_incomplete(self):
        key_list = [torch.device("cuda:10"), torch.device("cuda:0")]
        device_index_list = [0, 10, 13, 17]
        sorted_keys = _sort_key_list(key_list, device_index_list)
        for i in range(len(key_list)):
            self.assertEqual(sorted_keys[i].index, device_index_list[i])

    def test_select_target_2d(self) -> None:
        output_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assertTensorAlmostEqual(self, _select_targets(output_tensor, 1), [2, 5, 8])
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, torch.tensor(0)), [1, 4, 7]
        )
        assertTensorAlmostEqual(
            self,
            _select_targets(output_tensor, torch.tensor([1, 2, 0])),
            [[2], [6], [7]],
        )
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, [1, 2, 0]), [[2], [6], [7]]
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
