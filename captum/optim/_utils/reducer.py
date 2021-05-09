from typing import Any, Callable, List, Union

import numpy as np
import torch.nn.functional as F

try:
    import sklearn.decomposition
    from sklearn.base import BaseEstimator
except (ImportError, AssertionError):
    print(
        "The sklearn library is required to use Captum's ChannelReducer"
        + " unless you supply your own reduction algorithm."
    )
import torch


class ChannelReducer:
    """
    Dimensionality reduction for the channel dimension of an input.
    The default reduction_alg is NMF from sklearn, which requires users
    to put input on CPU before passing to fit_transform.
    Olah, et al., "The Building Blocks of Interpretability", Distill, 2018.
    See: https://distill.pub/2018/building-blocks/

    Args:
        n_components (int):  The number of channels to reduce the target
            dimension to.
        reduction_alg (str or callable):  The desired dimensionality
            reduction algorithm to use.
        **kwargs: Arbitrary keyword arguments used by the specified reduction_alg.
    """

    def __init__(
        self, n_components: int = 3, reduction_alg: Any = "NMF", **kwargs
    ) -> None:
        if isinstance(reduction_alg, str):
            reduction_alg = self._get_reduction_algo_instance(reduction_alg)
            if reduction_alg is None:
                raise ValueError(
                    "Unknown sklearn dimensionality reduction method '%s'."
                    % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)

    def _get_reduction_algo_instance(self, name: str) -> Union[None, Callable]:
        if hasattr(sklearn.decomposition, name):
            obj = sklearn.decomposition.__getattribute__(name)
            if issubclass(obj, BaseEstimator):
                return obj
        return None

    @classmethod
    def _apply_flat(cls, func: Callable, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        try:
            return func(x.reshape([-1, x.shape[-1]])).reshape(
                list(orig_shape[:-1]) + [-1]
            )
        except TypeError:
            raise TypeError(
                "The provided input is incompatible with the reduction_alg. "
                "Try placing the input on CPU first via x.cpu()."
            )

    def fit_transform(
        self, x: torch.Tensor, swap_2nd_and_last_dims: bool = True
    ) -> torch.Tensor:
        """
        Perform dimensionality reduction on an input tensor.

        Args:
            tensor (tensor):  A tensor to perform dimensionality reduction on.
            swap_2nd_and_last_dims (bool):   If true, input channels are expected
                to be in the second dimension unless the input tensor has a shape
                of CHW.
        Returns:
            *tensor*:  A tensor with one of it's dimensions reduced.
        """

        if x.dim() == 3 and swap_2nd_and_last_dims:
            x = x.permute(2, 1, 0)
        elif x.dim() > 3 and swap_2nd_and_last_dims:
            permute_vals = [0] + list(range(x.dim()))[2:] + [1]
            x = x.permute(*permute_vals)

        x_out = ChannelReducer._apply_flat(self._reducer.fit_transform, x)

        x_out = torch.as_tensor(x_out, device=x.device)

        if x.dim() == 3 and swap_2nd_and_last_dims:
            x_out = x_out.permute(2, 1, 0)
        elif x.dim() > 3 and swap_2nd_and_last_dims:
            permute_vals = (
                [0]
                + [x.dim() - 1]
                + list(range(x.dim()))[1 : len(list(range(x.dim()))) - 1]
            )
            x_out = x_out.permute(*permute_vals)

        return x_out

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            out = self.__dict__[name]
        elif name + "_" in self._reducer.__dict__:
            out = self._reducer.__dict__[name + "_"]
        if type(out) == np.ndarray:
            out = torch.as_tensor(out)
        return out

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

    Args:
        x (tensor):  A tensor to make positive.
        dim (int):  The dimension to concatinate the two tensor halves at.

    Returns:
        *tensor*:  A positive tensor for one-sided dimensionality reduction.
    """

    return torch.cat([F.relu(x), F.relu(-x)], dim=dim)


__all__ = [
    "ChannelReducer",
    "posneg",
]
