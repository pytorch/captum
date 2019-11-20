#!/usr/bin/env python3
from captum.attr._utils.stat import StatGraph


class Aggregator:
    """
    This class simply wraps over a given a set of Stat objects.

    Basic usage:

    >>>from captum.attr.aggregator import Aggregator
    >>>from captum.attr._utils.stats import Mean, StdDev
    >>>
    >>>attrib = torch.tensor([1, 2, 3, 4, 5])
    >>>
    >>>aggr = Aggregator([Mean, StdDev])
    >>>aggr.update(attrib)
    >>>
    >>>print(aggr.summary['mean'])
    """
    def __init__(self, stats=None):
        self._stat_graph = StatGraph()
        self.add_all(stats)

    def add_all(self, stats):
        """
        Adds a list of stat modules to this Aggregator
        """
        for stat in stats:
            self.add(stat)
        return self

    def add(self, stat):
        """
        Adds a stat module to this Aggregator. Please 
        note that this must be a module **not** an object, e.g.

        >>>aggr.add(Mean)
        """
        self._stat_graph.add(stat)
        return self

    def update(self, x):
        """ 
        Calls .update on each Stat object within this object
        """
        self._stat_graph.traverse(x)

    @property
    def summary(self):
        """ 
        Calls .get on each Stat object within this object

        Returns:
            A dict, mapping from the Stat object's .name to the associated value of .get
        """
        return self._stat_graph.summary


def common_aggr():
    from captum.aggr.stat import Mean, Var, StdDev, SampleStdDev, Min, Max

    return Aggregator([Mean, Var, StdDev, SampleStdDev, Min, Max])
