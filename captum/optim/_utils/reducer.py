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

    Olah, et al., "The Building Blocks of Interpretability", Distill, 2018.
    See: https://distill.pub/2018/building-blocks/
    """

    def __init__(
        self,
        n_components: int = 3,
        reduction_alg: Any = "NMF",
        supports_gpu: bool = False,
        **kwargs
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
        # Denotes whether _reducer is supports GPU inputs
        self.supports_gpu = supports_gpu

    def _get_reduction_algo_instance(self, name: str) -> Union[None, Callable]:
        if hasattr(sklearn.decomposition, name):
            obj = sklearn.decomposition.__getattribute__(name)
            if issubclass(obj, BaseEstimator):
                return obj
        return None

    @classmethod
    def _apply_flat(cls, func: Callable, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        return func(x.reshape([-1, x.shape[-1]])).reshape(list(orig_shape[:-1]) + [-1])

    def fit_transform(
        self, x: torch.Tensor, swap_2nd_and_last_dims: bool = True
    ) -> torch.Tensor:
        """
        Perform dimensionality reduction on an input tensor.

        If swap_2nd_and_last_dims is true, input channels are expected to be in the
        second dimension unless the input tensor has a shape of CHW.
        """

        if x.dim() == 3 and swap_2nd_and_last_dims:
            x = x.permute(2, 1, 0)
        elif x.dim() > 3 and swap_2nd_and_last_dims:
            permute_vals = [0] + list(range(x.dim()))[2:] + [1]
            x = x.permute(*permute_vals)

        if not self.supports_gpu:
            x_out = ChannelReducer._apply_flat(self._reducer.fit_transform, x.cpu())
        else:
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
    """

    return torch.cat([F.relu(x), F.relu(-x)], dim=dim)
