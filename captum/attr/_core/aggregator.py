#!/usr/bin/env python3
from captum.attr._utils.stat import StatGraph


class Aggregator:
    def __init__(self, stats=None):
        self._stat_graph = StatGraph()
        self.add_all(stats)

    def add_all(self, stats):
        for stat in stats:
            self.add(stat)
        return self

    def add(self, stat):
        self._stat_graph.add(stat)
        return self

    def update(self, x):
        self._stat_graph.traverse(x)

    @property
    def summary(self):
        return self._stat_graph.summary


def common_aggr():
    from captum.aggr.stat import Mean, Var, StdDev, SampleStdDev, Min, Max

    return Aggregator([Mean, Var, StdDev, SampleStdDev, Min, Max])
