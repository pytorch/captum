#!/usr/bin/env python3
import numpy as np
import torch
import random

from captum.attr._utils.stat import SummarizerSingleTensor, Mean, Var, StdDev
from .helpers.utils import BaseTest, assertTensorAlmostEqual


def get_values(n=100, lo=None, hi=None, integers=False):
    for _ in range(n):
        if integers:
            yield random.randint(lo, hi)
        else:
            yield random.random() * (hi - lo) + lo


class Test(BaseTest):
    def test_div0(self):
        summarizer = SummarizerSingleTensor([Var(), Mean()])
        summ = summarizer.summary
        self.assertIsNone(summ["mean"])
        self.assertIsNone(summ["variance"])

        summarizer.update(torch.tensor(10))
        summ = summarizer.summary
        assertTensorAlmostEqual(self, summ["mean"], 10)
        assertTensorAlmostEqual(self, summ["variance"], 0)

        summarizer.update(torch.tensor(10))
        summ = summarizer.summary
        assertTensorAlmostEqual(self, summ["mean"], 10)
        assertTensorAlmostEqual(self, summ["variance"], 0)

    def test_var_defin(self):
        """
        Variance is avg squared distance to mean. Thus it should be positive.
        This test is to ensure this is the case.

        To test it, we will we make a skewed distribution leaning to one end
        (either very large or small values).

        We will also compare to numpy and ensure it is approximately the same.
        This is assuming numpy is correct, for which it should be.
        """
        SMALL_VAL = -10000
        BIG_VAL = 10000
        AMOUNT_OF_SMALLS = [100, 10]
        AMOUNT_OF_BIGS = [10, 100]
        for sm, big in zip(AMOUNT_OF_SMALLS, AMOUNT_OF_BIGS):
            summ = SummarizerSingleTensor([Var()])
            values = []
            for i in range(sm):
                values.append(SMALL_VAL)
                summ.update(torch.tensor(SMALL_VAL, dtype=torch.float64))

            for i in range(big):
                values.append(BIG_VAL)
                summ.update(torch.tensor(BIG_VAL, dtype=torch.float64))

            actual_var = np.var(values)
            actual_var = torch.from_numpy(np.array(actual_var))

            var = summ.summary["variance"]

            assertTensorAlmostEqual(self, var, actual_var)
            self.assertTrue((var > 0).all())

    def test_stats_random_data(self):
        N = 1000
        BIG_VAL = 100000
        values = list(get_values(lo=-BIG_VAL, hi=BIG_VAL, n=N))
        stats_to_test = [Mean(), Var(), Var(order=1), StdDev(), StdDev(order=1)]
        stat_names = [
            "mean",
            "variance",
            "sample_variance",
            "std_dev",
            "sample_std_dev",
        ]
        gt_fns = [
            np.mean,
            np.var,
            lambda x: np.var(x, ddof=1),
            np.std,
            lambda x: np.std(x, ddof=1),
        ]

        for stat, name, gt in zip(stats_to_test, stat_names, gt_fns):
            summ = SummarizerSingleTensor([stat])
            for x in values:
                summ.update(torch.tensor(x, dtype=torch.float64))

            actual = torch.from_numpy(np.array(gt(values)))
            stat_val = summ.summary[name]

            assertTensorAlmostEqual(self, stat_val, actual)
