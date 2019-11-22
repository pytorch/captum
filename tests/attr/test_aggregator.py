#!/usr/bin/env python3
import torch

from captum.attr._core.aggregator import Aggregator, common_aggr
from .helpers.utils import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_single_input(self):
        aggr = common_aggr()
        for _ in range(10):
            attrs = torch.randn(10, 5, 5)
            aggr.update(attrs)

        summ = aggr.summary
        self.assertIsNotNone(summ)
        self.assertTrue(isinstance(summ, dict))


    def test_multi_input(self):
        aggr = common_aggr()
        for _ in range(10):
            a1 = torch.randn(10, 5, 5)
            a2 = torch.randn(10, 5, 5)
            aggr.update((a1, a2))

        summ = aggr.summary
        self.assertIsNotNone(summ)
        self.assertTrue(len(summ) == 2)
        self.assertTrue(isinstance(summ[0], dict))
        self.assertTrue(isinstance(summ[1], dict))
