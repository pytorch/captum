#!/usr/bin/env python3
import torch


class Stat:
    """
    Base class for Stat objects. You must call this constructor. 
    Sub-classes **must** have a default constructor.

    Args:
        deps (dict):
            The dependencies your Stat needs in order to perform and update and/or get.
            Maps from a string to a module, e.g. {'mean': Mean}.
    """
    def __init__(self, deps=None):
        if deps is None:
            deps = {}
        self.deps = deps

    def get(self, deps):
        """ 
        deps is a mapping from string to the associated value of the stat
        """
        raise NotImplementedError()

    def update(self, x, deps):
        """ 
        Args:
            x (torch.Tensor): 
                Some arbitrary tensor
            deps (dict): 
                A mapping from string to the associated value of the stat.
                This corresponds to the deps supplied in __init__.
        """
        raise NotImplementedError()

    @property
    def name(self):
        return "stat"


class Count(Stat):
    def __init__(self):
        super().__init__()
        self.n = None

    def get(self, deps):
        return self.n

    def update(self, x, deps):
        # TODO: figure out how
        #       to handle sparse input(s)
        if self.n is None:
            self.n = 0
        self.n += 1

    @property
    def name(self):
        return "count"


class Mean(Stat):
    def __init__(self):
        super().__init__({"n": Count})
        self.rolling_mean = None

    def get(self, deps):
        n = deps["n"]
        if n is None:
            return None
        return self.rolling_mean

    def update(self, x, deps):
        n = deps["n"]
        if self.rolling_mean is None:
            self.rolling_mean = x
        else:
            delta = x - self.rolling_mean
            self.rolling_mean += delta / n

    @property
    def name(self):
        return "mean"


class MSE(Stat):
    def __init__(self):
        super().__init__({"mean": Mean})
        self.prev_mean = None
        self.mse = None

    def get(self, deps):
        return self.mse

    def update(self, x, deps):
        mean = deps["mean"]
        if mean is not None and self.prev_mean is not None:
            rhs = (x - self.prev_mean) * (x - mean)
            if self.mse is None:
                self.mse = rhs
            else:
                self.mse += rhs

        # do not not clone
        self.prev_mean = mean.clone()

    @property
    def name(self):
        return "mse"


class Var(Stat):
    def __init__(self):
        super().__init__({"mse": MSE, "count": Count})

    def get(self, deps):
        mse = deps["mse"]
        n = deps["count"]
        return mse / n if mse is not None else None

    def update(self, x, deps):
        pass

    @property
    def name(self):
        return "variance"


class StdDev(Stat):
    def __init__(self):
        super().__init__({"var": Var})

    def get(self, deps):
        var = deps["var"]
        return var ** 0.5 if var is not None else None

    def update(self, x, deps):
        pass

    @property
    def name(self):
        return "std_dev"


class SampleVar(Stat):
    def __init__(self):
        super().__init__({"mse": MSE, "count": Count})

    def get(self, deps):
        mse = deps["mse"]
        n = deps["count"]
        if n - 1 <= 0 or mse is None:
            return None

        return mse / (n - 1)

    def update(self, x, deps):
        pass

    @property
    def name(self):
        return "sample_variance"


class SampleStdDev(Stat):
    def __init__(self):
        super().__init__({"var": SampleVar})

    def get(self, deps):
        # TODO: be DRY
        var = deps["var"]
        return var ** 0.5 if var is not None else None

    def update(self, x, deps):
        pass

    @property
    def name(self):
        return "sample_std_dev"


class GeneralAccumFn(Stat):
    def __init__(self, fn):
        super().__init__()
        self.result = None
        self.fn = fn

    def get(self, deps):
        return self.result

    def update(self, x, deps):
        if self.result is None:
            self.result = x
        else:
            self.result = self.fn(self.result, x)


class Min(GeneralAccumFn):
    def __init__(self, min_fn=torch.min):
        super().__init__(fn=min_fn)

    @property
    def name(self):
        return "min"


class Max(GeneralAccumFn):
    def __init__(self, max_fn=torch.max):
        super().__init__(fn=max_fn)

    @property
    def name(self):
        return "max"


class StatGraph:
    class Node:
        stat = None
        invisible = False

        def __init__(self, stat, invisible):
            self.stat = stat
            self.invisible = invisible

    def __init__(self):
        self.is_ready = False
        self.module_to_node = dict()
        self.nodes = []

    def add(self, stat, invisible=False):
        if stat in self.module_to_node:
            self.module_to_node[stat].invisible = False
            return self

        self.is_ready = False
        node = StatGraph.Node(stat=stat(), invisible=invisible)
        self.nodes.append(node)
        self.module_to_node[stat] = node
        return self

    def iter_all(self, visible=True):
        for node in self.nodes:
            if visible and node.invisible:
                continue

            yield node.stat

    def contains(self, stat_module):
        return stat_module in self.module_to_node

    def _resolve_deps(self):
        unsat = False
        for stat in self.iter_all(visible=False):
            for name, dep in stat.deps.items():
                if not self.contains(dep):
                    self.add(dep, invisible=True)
                    unsat = True

        if unsat:
            self._resolve_deps()
        else:

            def get_parents(node):
                for _, dep in node.stat.deps.items():
                    yield self.module_to_node[dep]

            self.nodes = list(_topo_sort(self.nodes, get_parents))
            self.is_ready = True

    def traverse(self, x=None):
        if not self.is_ready:
            self._resolve_deps()

        summ = {}
        for stat in self.iter_all(visible=False):
            deps = {}

            for name, module in stat.deps.items():
                assert module in summ
                deps[name] = summ[module]

            if x is not None:
                stat.update(x, deps)

            summ[stat.__class__] = stat.get(deps)

        return summ

    @property
    def summary(self):
        summ = self.traverse()

        return {
            self.module_to_node[k].stat.name: v
            for k, v in summ.items()
            if not self.module_to_node[k].invisible
        }


def _dfs(node, parent_map, visited, marked):
    marked.add(node)
    for parent in parent_map(node):
        if parent in visited:
            continue
        if parent in marked:
            return None

        for x in _dfs(parent, parent_map, visited, marked):
            yield x

    visited.add(node)
    yield node


def _topo_sort(nodes, parent_map=None):
    visited = set()
    order = []
    for node in nodes:
        if node in visited:
            continue

        marked = set()
        for node in _dfs(node, parent_map, visited, marked):
            if node is None:
                return None

            order.append(node)

    return order
