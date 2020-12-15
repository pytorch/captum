from typing import Any, Callable, List, Union

import numpy as np

try:
    import sklearn.decomposition
    from sklearn.base import BaseEstimator
except (ImportError, AssertionError):
    print(
        "The sklearn library is required to use Captum's ChannelReducer"
        + " unless you supply your own reduction algorithm."
    )
import torch


class ChannelReducer(object):
    """
    Dimensionality reduction for the channel dimension of an input.

    Olah, et al., "The Building Blocks of Interpretability", Distill, 2018.
    See: https://distill.pub/2018/building-blocks/
    """

    def __init__(
        self, n_components: int = 3, reduction_alg: Any = "NMF", **kwargs
    ) -> None:
        if not callable(reduction_alg):
            algorithm_map = {}
            for name in dir(sklearn.decomposition):
                obj = sklearn.decomposition.__getattribute__(name)
                if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                    algorithm_map[name] = obj
            if isinstance(reduction_alg, str):
                if reduction_alg in algorithm_map:
                    reduction_alg = algorithm_map[reduction_alg]
                else:
                    raise ValueError(
                        "Unknown sklearn dimensionality reduction method '%s'."
                        % reduction_alg
                    )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)

    @classmethod
    def _apply_flat(cls, func: Callable, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        return func(x.reshape([-1, x.shape[-1]])).reshape(list(orig_shape[:-1]) + [-1])

    def fit_transform(
        self, x: Union[torch.Tensor, np.ndarray], reshape: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform dimensionality reduction on an input tensor or NumPy array.
        """

        is_tensor = torch.is_tensor(x)

        if is_tensor:
            if x.dim() == 3 and reshape:
                x = x.permute(2, 1, 0)
            elif x.dim() > 3 and reshape:
                permute_vals = [0] + list(range(x.dim()))[2:] + [1]
                x = x.permute(*permute_vals)
        else:
            if x.ndim == 3 and reshape:
                x = x.transpose(2, 1, 0)
            elif x.ndim > 3 and reshape:
                permute_vals = [0] + list(range(x.ndim))[2:] + [1]
                x = x.transpose(*permute_vals)

        x_out = ChannelReducer._apply_flat(self._reducer.fit_transform, x)

        if is_tensor:
            x_out = torch.as_tensor(x_out)
            if x.dim() == 3 and reshape:
                x_out = x_out.permute(2, 1, 0)
            elif x.dim() > 3 and reshape:
                permute_vals = (
                    [0]
                    + [x.dim() - 1]
                    + list(range(x.dim()))[1 : len(list(range(x.dim()))) - 1]
                )
                x_out = x_out.permute(*permute_vals)
        else:
            if x.ndim == 3 and reshape:
                x_out = x_out.permute(2, 1, 0)
            elif x.ndim > 3 and reshape:
                permute_vals = (
                    [0]
                    + [x.ndim - 1]
                    + list(range(x.ndim))[1 : len(list(range(x.ndim))) - 1]
                )
                x_out = x_out.transpose(*permute_vals)

        return x_out

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name + "_" in self._reducer.__dict__:
            return self._reducer.__dict__[name + "_"]

    def __dir__(self) -> List:
        dynamic_attrs = [
            name[:-1]
            for name in dir(self._reducer)
            if name[-1] == "_" and name[0] != "_"
        ]

        return (
            list(ChannelReducer.__dict__.keys())
            + list(self.__dict__.keys())
            + dynamic_attrs
        )


def posneg(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Hack that makes a matrix positive by concatination in order to simulate
    one-sided NMF with regular NMF
    """

    return torch.cat(
        [torch.max(x, torch.full_like(x, 0)), torch.max(-x, torch.full_like(x, 0))],
        dim=dim,
    )
