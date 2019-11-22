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
        self._stat_list = []
        self._stat_graphs = []
        self._is_inputs_tuple = None

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
        self._stat_list.append(stat)
        for graph in self._stat_graphs:
            graph.add(stat)
        return self

    def update(self, x):
        """
        Calls .update on each Stat object within this object
        """
        if self._is_inputs_tuple is None:
            self._is_inputs_tuple = isinstance(x, tuple)
        else:
            # we want input to be consistently a single input or a tuple
            assert not (self._is_inputs_tuple ^ isinstance(x, tuple))

        if self._stat_graphs is None:
            self._stat_graphs = []

        for i, inp in enumerate(x):
            if i >= len(self._stat_graphs):
                self._stat_graphs.append(StatGraph(stats=self._stat_list))
            self._stat_graphs[i].traverse(inp)

    @property
    def summary(self):
        """
        Calls .get on each Stat object within this object

        Returns:
            A dict, mapping from the Stat object's .name to the associated value of .get
        """
        out = [graph.summary for graph in self._stat_graphs]
        if self._is_inputs_tuple:
            return tuple(out)

        return out[0]


def common_aggr():
    from captum.attr._utils.stat import Mean, Var, StdDev, SampleStdDev, Min, Max

    return Aggregator([Mean, Var, StdDev, SampleStdDev, Min, Max])
