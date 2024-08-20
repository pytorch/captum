#!/usr/bin/env python3

# pyre-strict
from typing import Any, Callable, List, Optional, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from captum.attr._utils.summarizer import SummarizerSingleTensor


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

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Args:
            name (str, optional):
                The name of the statistic. If not provided,
                the class name will be used alongside it's parameters
            kwargs (Any):
                Additional arguments used to construct the statistic
        """
        # pyre-fixme[4]: Attribute must be annotated.
        self.params = kwargs
        self._name = name

        self._other_stats: Optional[SummarizerSingleTensor] = None

    def init(self) -> None:
        pass

    def _get_stat(self, stat: "Stat") -> Optional["Stat"]:
        assert self._other_stats is not None
        return self._other_stats.get(stat)

    # pyre-fixme[3]: Return type must be annotated.
    def update(self, x: Tensor):
        raise NotImplementedError()

    def get(self) -> Optional[Tensor]:
        raise NotImplementedError()

    def __hash__(self) -> int:
        return hash((self.__class__, frozenset(self.params.items())))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Stat):
            return self.__class__ == other.__class__ and frozenset(
                self.params.items()
            ) == frozenset(other.params.items())
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def name(self) -> str:
        """
        The name of the statistic. i.e. it is the key in a .summary

        This will be the class name or a custom name if provided.

        See Summarizer or SummarizerSingleTensor
        """
        default_name = self.__class__.__name__.lower()
        if len(self.params) > 0:
            default_name += f"({self.params})"

        return default_name if self._name is None else self._name


class Count(Stat):
    """
    Counts the number of elements, i.e. the
    number of `update`'s called
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n: Optional[int] = None

    # pyre-fixme[15]: `captum.attr._utils.stat.Count.get` overrides method defined
    # in `Stat` inconsistently. Returned type `Optional[int]` is not a subtype of
    # the overridden return `Optional[torch._tensor.Tensor]`.
    def get(self) -> Optional[int]:  # type: ignore
        return self.n

    def update(self, x: Tensor) -> None:
        if self.n is None:
            self.n = 0
        self.n += 1


class Mean(Stat):
    """
    Calculates the average of a tensor
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.rolling_mean: Optional[Tensor] = None
        self.n: Optional[Count] = None

    def get(self) -> Optional[Tensor]:
        return self.rolling_mean

    def init(self) -> None:
        # pyre-fixme[8]: Attribute has type `Optional[Count]`; used as `Optional[Stat]`.
        self.n = self._get_stat(Count())  # type: ignore

    def update(self, x: Tensor) -> None:
        # pyre-fixme[16]: `Optional` has no attribute `get`.
        n = self.n.get()  # type: ignore

        if self.rolling_mean is None:
            # Ensures rolling_mean is a float tensor
            self.rolling_mean = x.clone() if x.is_floating_point() else x.double()
        else:
            delta = x - self.rolling_mean
            # pyre-fixme[16]: `Optional` has no attribute `__iadd__`.
            self.rolling_mean += delta / n


class MSE(Stat):
    """
    Calculates the mean squared error of a tensor
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        # pyre-fixme[4]: Attribute must be annotated.
        self.prev_mean = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.mse = None

    def init(self) -> None:
        # pyre-fixme[16]: `MSE` has no attribute `mean`.
        self.mean = self._get_stat(Mean())

    def get(self) -> Optional[Tensor]:
        if self.mse is None and self.prev_mean is not None:
            return torch.zeros_like(self.prev_mean)
        return self.mse

    def update(self, x: Tensor) -> None:
        # pyre-fixme[16]: `MSE` has no attribute `mean`.
        mean = self.mean.get()  # type: ignore

        if mean is not None and self.prev_mean is not None:
            rhs = (x - self.prev_mean) * (x - mean)
            if self.mse is None:
                self.mse = rhs
            else:
                self.mse += rhs

        # do not not clone
        self.prev_mean = mean.clone()  # type: ignore


class Var(Stat):
    """
    Calculates the variance of a tensor, with an order. e.g.
    if `order = 1` then it will calculate sample variance.

    This is equal to mse / (n - order)
    """

    def __init__(self, name: Optional[str] = None, order: int = 0) -> None:
        if name is None:
            if order == 0:
                name = "variance"
            elif order == 1:
                name = "sample_variance"
            else:
                name = f"variance({order})"

        super().__init__(name=name, order=order)
        self.order = order

    def init(self) -> None:
        # pyre-fixme[16]: `Var` has no attribute `mse`.
        self.mse = self._get_stat(MSE())
        # pyre-fixme[16]: `Var` has no attribute `n`.
        self.n = self._get_stat(Count())

    def update(self, x: Tensor) -> None:
        pass

    def get(self) -> Optional[Tensor]:
        # pyre-fixme[16]: `Var` has no attribute `mse`.
        mse = self.mse.get()  # type: ignore
        # pyre-fixme[16]: `Var` has no attribute `n`.
        n = self.n.get()  # type: ignore

        if mse is None:
            return None

        if n <= self.order:  # type: ignore
            return torch.zeros_like(mse)

        # NOTE: The following ensures mse is a float tensor.
        #   torch.true_divide is available in PyTorch 1.5 and later.
        #   This is for compatibility with 1.4.
        return mse.to(torch.float64) / (n - self.order)  # type: ignore


class StdDev(Stat):
    """
    The standard deviation, with an associated order.
    """

    def __init__(self, name: Optional[str] = None, order: int = 0) -> None:
        if name is None:
            if order == 0:
                name = "std_dev"
            elif order == 1:
                name = "sample_std_dev"
            else:
                name = f"std_dev{order})"

        super().__init__(name=name, order=order)
        self.order = order

    def init(self) -> None:
        # pyre-fixme[16]: `StdDev` has no attribute `var`.
        self.var = self._get_stat(Var(order=self.order))

    def update(self, x: Tensor) -> None:
        pass

    def get(self) -> Optional[Tensor]:
        # pyre-fixme[16]: `StdDev` has no attribute `var`.
        var = self.var.get()  # type: ignore
        return var**0.5 if var is not None else None


class GeneralAccumFn(Stat):
    """
    Performs update(x): result = fn(result, x)
    where fn is a custom function
    """

    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def __init__(self, fn: Callable, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.result: Optional[Tensor] = None
        self.fn = fn

    def get(self) -> Optional[Tensor]:
        return self.result

    def update(self, x: Tensor) -> None:
        if self.result is None:
            self.result = x
        else:
            self.result = self.fn(self.result, x)


class Min(GeneralAccumFn):
    def __init__(
        self,
        name: Optional[str] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        min_fn: Callable = torch.min,
    ) -> None:
        super().__init__(name=name, fn=min_fn)


class Max(GeneralAccumFn):
    def __init__(
        self,
        name: Optional[str] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        max_fn: Callable = torch.max,
    ) -> None:
        super().__init__(name=name, fn=max_fn)


class Sum(GeneralAccumFn):
    def __init__(
        self,
        name: Optional[str] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        add_fn: Callable = torch.add,
    ) -> None:
        super().__init__(name=name, fn=add_fn)


def CommonStats() -> List[Stat]:
    r"""
    Returns common summary statistics, specifically:
        Mean, Sample Variance, Sample Std Dev, Min, Max
    """
    return [Mean(), Var(order=1), StdDev(order=1), Min(), Max()]
