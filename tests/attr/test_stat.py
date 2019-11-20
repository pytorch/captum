#!/usr/bin/env python3
import numpy as np
import torch
import random

from captum.attr._utils.stat import (
    StatGraph,
    Mean,
    MSE,
    Var,
    StdDev,
    SampleStdDev,
    SampleVar,
    Min,
    Max,
    _topo_sort,
)
from .helpers.utils import BaseTest, assertTensorAlmostEqual


def get_values(n=100, lo=None, hi=None, integers=False):
    for _ in range(n):
        if integers:
            yield random.randint(lo, hi)
        else:
            yield random.random() * (hi - lo) + lo


class Test(BaseTest):
    def test_div0(self):
        stat_graph = StatGraph().add(Var).add(Mean)
        summ = stat_graph.summary
        self.assertIsNone(summ["mean"])
        self.assertIsNone(summ["variance"])

        stat_graph.traverse(torch.tensor(10))
        summ = stat_graph.summary
        assertTensorAlmostEqual(self, summ["mean"], 10)
        self.assertIsNone(summ["variance"])

        stat_graph.traverse(torch.tensor(10))
        summ = stat_graph.summary
        assertTensorAlmostEqual(self, summ["mean"], 10)
        assertTensorAlmostEqual(self, summ["variance"], 0)

    def test_var_defin(self):
        """ 
        Variance is avg squared distance to mean. Thus it should be positive. This test 
        is to ensure this is the case.

        To test it, we will we make a skewed distribution leaning to one end (either very 
        large or small values).

        We will also compare to numpy and ensure it is approximately the same. This 
        is assuming numpy is correct, for which it should be.
        """
        SMALL_VAL = -10000
        BIG_VAL = 10000
        AMOUNT_OF_SMALLS = [100, 10]
        AMOUNT_OF_BIGS = [10, 100]
        for sm, big in zip(AMOUNT_OF_SMALLS, AMOUNT_OF_BIGS):
            graph = StatGraph().add(Var)
            values = []
            for i in range(sm):
                values.append(SMALL_VAL)
                graph.traverse(torch.tensor(SMALL_VAL, dtype=torch.float64))

            for i in range(big):
                values.append(BIG_VAL)
                graph.traverse(torch.tensor(BIG_VAL, dtype=torch.float64))

            actual_var = np.var(values)
            actual_var = torch.from_numpy(np.array(actual_var))

            var = graph.summary["variance"]

            assertTensorAlmostEqual(self, var, actual_var)
            self.assertTrue((var > 0).all())

    def test_stats_random_data(self):
        N = 1000
        BIG_VAL = 100000
        values = list(get_values(lo=-BIG_VAL, hi=BIG_VAL, n=N))
        stats_to_test = [Mean, Var, SampleVar, StdDev, SampleStdDev]
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
            graph = StatGraph().add(stat)
            for x in values:
                graph.traverse(torch.tensor(x, dtype=torch.float64))

            actual = torch.from_numpy(np.array(gt(values)))
            stat_val = graph.summary[name]

            assertTensorAlmostEqual(self, stat_val, actual)

    def test_topo_sort_random(self):
        N = 100
        max_num_nodes = 100
        min_num_nodes = 1
        # create N random graphs
        for i in range(N):
            # each node is represented by an integer i, 0 <= i < num_nodes
            num_nodes = random.randint(min_num_nodes, max_num_nodes)

            # let's actually create the nodes of the graph
            # i.e. the first node is represented as nodes[0]
            #
            # thus the ordering might not necessarily be the integers sorted
            nodes = np.random.permutation(num_nodes).tolist()

            # let's construct our graph by
            # randomly assigning back edges
            edges = [None] * num_nodes
            for i in range(num_nodes):
                if i == 0:
                    edges[nodes[i]] = set()
                    continue

                num_edges = random.randint(0, num_nodes - i - 1)
                e = set()
                for j in range(num_edges):
                    to_node_idx = random.randint(0, i - 1)
                    e.add(nodes[to_node_idx])

                edges[nodes[i]] = e

            def parent_map(node):
                return edges[node]

            # topo sort should work independent of ordering given to algo
            random_node_order = np.random.permutation(num_nodes)
            order = _topo_sort(random_node_order, parent_map)
            self.assertIsNotNone(order)

            seen = set()
            for idx in order:
                node = idx
                for parent in edges[node]:
                    self.assertTrue(parent in seen)

                seen.add(node)
