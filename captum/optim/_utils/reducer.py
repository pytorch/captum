from typing import Callable

import sklearn.decomposition
import torch


class ChannelReducer(object):
    """
    Reduce the channel size of activations to a more visualization friendly value.
    """

    def __init__(self, n_components: int = 3, **kwargs) -> None:
        self.n_components = n_components
        self._reducer = sklearn.decomposition._nmf.NMF(
            n_components=n_components, **kwargs
        )

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
