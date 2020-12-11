from typing import Any, Callable

try:
    import sklearn.decomposition
    from sklearn.base import BaseEstimator
except (ImportError, AssertionError):
    print("The sklearn library is required to use Captum's ChannelReducer")
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
                    "Unknown dimensionality reduction method '%s'." % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)

    @classmethod
    def _apply_flat(cls, func: Callable, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        return func(x.reshape([-1, x.shape[-1]])).reshape(list(orig_shape[:-1]) + [-1])

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move channels to channels last NumPy format
        Assume first dimension is batch size except for shape CHW
        """

        if x.dim() == 3:
            x = x.permute(2, 1, 0)
        elif x.dim() > 3:
            permute_vals = [0] + list(range(x.dim()))[2:] + [1]
            x = x.permute(*permute_vals)

        x_out = torch.as_tensor(
            ChannelReducer._apply_flat(self._reducer.fit_transform, x)
        )

        if x.dim() == 3:
            x_out = x_out.permute(2, 1, 0)
        elif x.dim() > 3:
            permute_vals = (
                [0]
                + [x.dim() - 1]
                + list(range(x.dim()))[1 : len(list(range(x.dim()))) - 1]
            )
            x_out = x_out.permute(*permute_vals)
        return x_out
