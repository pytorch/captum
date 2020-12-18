import math
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
            x_out = torch.as_tensor(x_out, device=x.device)
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

    return torch.cat(
        [torch.max(x, torch.full_like(x, 0)), torch.max(-x, torch.full_like(x, 0))],
        dim=dim,
    )


def nchannels_to_rgb(x: torch.Tensor, warp: bool = True) -> torch.Tensor:
    """
    Convert an NCHW image with n channels into a 3 channel RGB image.

    Args:
        x (torch.Tensor):  Image tensor to transform into RGB image.
        warp (bool, optional):  Whether or not to make colors more distinguishable.
            Default: True
    Returns:
        *tensor* RGB image
    """

    def hue_to_rgb(angle: float) -> torch.Tensor:
        """
        Create an RGB unit vector based on a hue of the input angle.
        """

        angle = angle - 360 * (angle // 360)
        colors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.7071, 0.7071, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.7071, 0.7071],
                [0.0, 0.0, 1.0],
                [0.7071, 0.0, 0.7071],
            ]
        )

        idx = math.floor(angle / 60)
        d = (angle - idx * 60) / 60

        if warp:

            def adj(x: float) -> float:
                return math.sin(x * math.pi / 2)

            d = adj(d) if idx % 2 == 0 else 1 - adj(1 - d)

        vec = (1 - d) * colors[idx] + d * colors[(idx + 1) % 6]
        return vec / torch.norm(vec)

    assert x.dim() == 4

    if (x < 0).any():
        x = posneg(x.permute(0, 2, 3, 1), -1).permute(0, 3, 1, 2)

    rgb = torch.zeros(1, 3, x.size(2), x.size(3), device=x.device)
    nc = x.size(1)
    for i in range(nc):
        rgb = rgb + x[:, i][:, None, :, :]
        rgb = rgb * hue_to_rgb(360 * i / nc).to(device=x.device)[None, :, None, None]

    rgb = rgb + torch.ones(x.size(2), x.size(3))[None, None, :, :] * (
        torch.sum(x, 1)[:, None] - torch.max(x, 1)[0][:, None]
    )
    return (rgb / (1e-4 + torch.norm(rgb, dim=1, keepdim=True))) * torch.norm(
        x, dim=1, keepdim=True
    )
