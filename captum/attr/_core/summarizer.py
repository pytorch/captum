#!/usr/bin/env python3
import torch


class Summarizer:
    """
    This class simply wraps over a given a set of SummarizerSingleTenor's in order
    to summarise multiple input tensors.

    Basic usage:

    >>>from captum.attr.aggregator import Summarizer
    >>>from captum.attr._utils.stats import Mean, StdDev
    >>>
    >>>attrib = torch.tensor([1, 2, 3, 4, 5])
    >>>
    >>>summ = Summarizer([Mean(), StdDev(0])
    >>>summ.update(attrib)
    >>>
    >>>print(summ.summary['mean'])
    """

    def __init__(self, stats=None):
        self._summarizers = []
        self._is_inputs_tuple = None
        self._stats = stats

    def _copy_stats(self):
        import copy

        return copy.deepcopy(self._stats)

    def update(self, x):
        """
        Calls .update on each Stat object within this object
        """
        from captum.attr._utils.stat import SummarizerSingleTensor

        if self._is_inputs_tuple is None:
            self._is_inputs_tuple = isinstance(x, tuple)
        else:
            # we want input to be consistently a single input or a tuple
            assert not (self._is_inputs_tuple ^ isinstance(x, tuple))

        if not self._is_inputs_tuple:
            x = (x,)

        for i, inp in enumerate(x):
            while i >= len(self._summarizers):
                self._summarizers.append(
                    SummarizerSingleTensor(stats=self._copy_stats())
                )
            self._summarizers[i].update(inp)

    @property
    def summary(self):
        """
        Effectively calls .get on each Stat object within this object for each input

        Returns:
            A dict, mapping from the Stat object's .name to the associated value of .get
        """
        if len(self._summarizers) == 0:
            return {}

        temp = [summ.summary for summ in self._summarizers]
        return temp if self._is_inputs_tuple else temp[0]


"""
Returns a summarizer with common summary statistics, specifically with:
    Mean, Sample Variance, Sample Std Dev, Min, Max
"""


def CommonSummarizer():
    from captum.attr._utils.stat import Mean, Var, StdDev, Min, Max

    return Summarizer([Mean(), Var(order=1), StdDev(order=1), Min(), Max()])
