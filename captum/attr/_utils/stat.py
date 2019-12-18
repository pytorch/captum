#!/usr/bin/env python3
import torch


class Stat:
    """
    The Stat class represents a statistic that can be updated and retrieved
    at any point in time. 

    The basic functionality this class provides is:
    1. A update/get method to actually compute the statistic
    2. A statistic store/cache to retrieve dependent information 
       (e.g. other stat values that are required for computation)
    3. The name of the statistic that is used for the user to refer to
    """
    def __init__(self, name=None, **kwargs):
        self.params = kwargs
        self._name = name

        self._other_stats = None

    def init(self):
        pass

    def _get_stat(self, stat):
        return self._other_stats.get(stat)

    def update(self, x):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash((self.__class__, frozenset(self.params.items())))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               frozenset(self.params.items()) == frozenset(other.params.items())

    def __ne__(self, other):
        return not self.__eq__(other)

    """
    The name of the statistic. i.e. it is the key in a .summary 

    See Summarizer or SummarizerSingleTensor
    """
    @property
    def name(self):
        default_name = self.__class__.__name__.lower()
        if len(self.params) > 0:
            default_name += f'({self.params})'

        return default_name if self._name is None else self._name

class Count(Stat):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.n = None

    def get(self):
        return self.n

    def update(self, x):
        if self.n is None:
            self.n = 0
        self.n += 1

class Mean(Stat):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.rolling_mean = None
        self.n = None

    def get(self):
        return self.rolling_mean

    def init(self):
        self.n = self._get_stat(Count())

    def update(self, x):
        n = self.n.get()

        if self.rolling_mean is None:
            self.rolling_mean = x
        else:
            delta = x - self.rolling_mean
            self.rolling_mean += delta / n

class MSE(Stat):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.prev_mean = None
        self.mse = None

    def init(self):
        self.mean = self._get_stat(Mean())

    def get(self):
        if self.mse is None and self.prev_mean is not None:
            return torch.zeros_like(self.prev_mean)
        return self.mse

    def update(self, x):
        mean = self.mean.get()

        if mean is not None and self.prev_mean is not None:
            rhs = (x - self.prev_mean) * (x - mean)
            if self.mse is None:
                self.mse = rhs
            else:
                self.mse += rhs

        # do not not clone
        self.prev_mean = mean.clone()

class Var(Stat):
    def __init__(self, name=None, order=0):
        if name is None:
            if order == 0:
                name = 'variance'
            elif order == 1:
                name = 'sample_variance'
            else:
                name = f'variance({order})'

        super().__init__(name=name, order=order)
        self.order = order

    def init(self):
        self.mse = self._get_stat(MSE())
        self.n = self._get_stat(Count())

    def update(self, x):
        pass

    def get(self):
        mse = self.mse.get()
        n = self.n.get()

        if mse is None:
            return None

        if n <= self.order:
            return torch.zeros_like(mse)

        return mse / (n - self.order)


class StdDev(Stat):
    def __init__(self, name=None, order=0):
        if name is None:
            if order == 0:
                name = 'std_dev'
            elif order == 1:
                name = 'sample_std_dev'
            else:
                name = f'std_dev{order})'

        super().__init__(name=name, order=order)
        self.order = order

    def init(self):
        self.var = self._get_stat(Var(order=self.order))

    def update(self, x):
        pass

    def get(self):
        var = self.var.get()
        return var ** 0.5 if var is not None else None

class GeneralAccumFn(Stat):
    def __init__(self, fn, name=None):
        super().__init__(name=name)
        self.result = None
        self.fn = fn

    def get(self):
        return self.result

    def update(self, x):
        if self.result is None:
            self.result = x
        else:
            self.result = self.fn(self.result, x)


class Min(GeneralAccumFn):
    def __init__(self, name=None, min_fn=torch.min):
        super().__init__(name=name, fn=min_fn)


class Max(GeneralAccumFn):
    def __init__(self, name=None, max_fn=torch.max):
        super().__init__(name=name, fn=max_fn)

class SummarizerSingleTensor:
    """
    A simple class that summarizes a single tensor. The basic functionality
    of this class is two operations .update and .summary
    """

    class StatHolder:
        def __init__(self, stats):
            # We want to want to store two things:
            # 1. A mapping from a Stat to Stat object (self._stat_to_stat):
            #    This is to retrieve an existing Stat object for dependency resolution, e.g.
            #    Mean needs the Count stat - we want to retrieve it in O(1)
            # 
            # 2. All of the necessary stats, in the correct order, 
            #    to perform an update for each Stat (self.stats) trivially

            # As a reference, the dependency graph for our stats is as follows:
            # StdDev(x) -> Var(x) -> MSE -> Mean -> Count, for all valid x
            #
            # Step 1: 
            #    Ensure we have all the necessary stats 
            #    i.e. ensure we have the dependencies
            # Step 2: 
            #    Figure out the order to update them
            dep_order = [StdDev, Var, MSE, Mean, Count]

            stats = set(stats) # remove dupe stats

            from collections import defaultdict
            stats_by_module = defaultdict(list)
            for stat in stats:
                stats_by_module[stat.__class__].append(stat)

            # StdDev is an odd case since it is parameterized, thus
            # for each StdDev(order) we must ensure there is an associated Var(order)
            for std_dev in stats_by_module[StdDev]:
                stat_to_add = Var(order=std_dev.order)
                stats.add(stat_to_add)
                stats_by_module[stat_to_add.__class__].append(stat_to_add)

            # For the other modules (deps[1:n-1]), if i exists => we want to ensure i...n-1 exists
            for i, dep in enumerate(dep_order[1:]):
                if dep in stats_by_module:
                    stats.update([mod() for mod in dep_order[i+1:]])
                    break

            # Step 2: get the correct order
            # NOTE: we are sorting via a given topological order
            sort_order = { mod: i for i, mod in enumerate(dep_order) }
            sort_order[Min] = -1
            sort_order[Max] = -1

            stats = list(stats)
            stats.sort(key=lambda x: sort_order[x.__class__], reverse=True)

            self.stats = stats
            self.stat_to_stat = {stat: stat for stat in self.stats}

            for stat in stats:
                stat._other_stats = self
                stat.init()

        def get(self, stat):
            if not stat in self.stat_to_stat:
                return None

            return self.stat_to_stat[stat]

    def __init__(self, stats=None):
        self._all_stats = SummarizerSingleTensor.StatHolder(stats)

        # this is what we actually want to output
        self._summary_stats = stats 

    def update(self, x=None):
        for stat in self._all_stats.stats:
            stat.update(x)

    @property
    def summary(self):
        return {stat.name: stat.get() for stat in self._summary_stats}
