#!/usr/bin/env python3

import torch
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class HelpersTest(BaseTest):
    def test_assert_tensor_almost_equal(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            assertTensorAlmostEqual(self, [[1.0]], [[1.0]])
        self.assertEqual(
            cm.exception.args,
            ("Actual parameter given for comparison must be a tensor.",),
        )

        with self.assertRaises(AssertionError) as cm:
            assertTensorAlmostEqual(self, torch.tensor([[]]), torch.tensor([[1.0]]))
        self.assertEqual(
            cm.exception.args,
            (
                "Expected tensor with shape: torch.Size([1, 1]). Actual shape torch.Size([1, 0]).",  # noqa: E501
            ),
        )

        assertTensorAlmostEqual(self, torch.tensor([[1.0]]), [[1.0]])

        with self.assertRaises(AssertionError) as cm:
            assertTensorAlmostEqual(self, torch.tensor([[1.0]]), [1.0])
        self.assertEqual(
            cm.exception.args,
            (
                "Expected tensor with shape: torch.Size([1]). Actual shape torch.Size([1, 1]).",  # noqa: E501
            ),
        )

        assertTensorAlmostEqual(
            self, torch.tensor([[1.0, 1.0]]), [[1.0, 0.0]], delta=1.0, mode="max"
        )

        with self.assertRaises(AssertionError) as cm:
            assertTensorAlmostEqual(
                self, torch.tensor([[1.0, 1.0]]), [[1.0, 0.0]], mode="max"
            )
        self.assertEqual(
            cm.exception.args,
            (
                "Values at index 0, tensor([1., 1.]) and tensor([1., 0.]), differ more than by 0.0001",  # noqa: E501
            ),
        )

        assertTensorAlmostEqual(
            self, torch.tensor([[1.0, 1.0]]), [[1.0, 0.0]], delta=1.0
        )

        with self.assertRaises(AssertionError):
            assertTensorAlmostEqual(self, torch.tensor([[1.0, 1.0]]), [[1.0, 0.0]])
