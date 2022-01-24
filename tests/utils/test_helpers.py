#!/usr/bin/env python3

import torch
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class HelpersTest(BaseTest):
    def test_assert_tensor_almost_equal(self) -> None:
        with self.assertRaises(AssertionError):
            assertTensorAlmostEqual(self, torch.tensor([[]]), torch.tensor([[1.0]]))

        assertTensorAlmostEqual(self, torch.tensor([[1.0]]), [[1.0]])
        with self.assertRaises(AssertionError):
            assertTensorAlmostEqual(self, torch.tensor([[1.0]]), [1.0])
