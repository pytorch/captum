import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.optim._core.output_hook import ActivationFetcher
from captum.optim._utils.typing import ModelInputType, ModuleOutputMapping


def get_model_layers(model) -> List[str]:
    """
    Return a list of hookable layers for the target model.
    """
    layers = []

    def get_layers(net, prefix: List = []) -> None:
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    continue
                separator = "" if str(name).isdigit() else "."
                name = "[" + str(name) + "]" if str(name).isdigit() else name
                layers.append(separator.join(prefix + [name]))
                get_layers(layer, prefix=prefix + [name])

    get_layers(model)
    return layers


class RedirectedReLU(torch.autograd.Function):
    """
    A workaround when there is no gradient flow from an initial random input.
    ReLU layers will block the gradient flow during backpropagation when their
    input is less than 0. This means that it can be impossible to visualize a
    target without allowing negative values to pass through ReLU layers during
    backpropagation.
    See:
    https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
    """

    @staticmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = grad_input[input_tensor < 0] * 1e-1
        return grad_input


class RedirectedReluLayer(nn.Module):
    """
    Class for applying RedirectedReLU
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if F.relu(input.detach().sum()) != 0:
            return F.relu(input, inplace=True)
        else:
            return RedirectedReLU.apply(input)


class ReluLayer(nn.Module):
    """
    Basic Hookable & Replaceable ReLU layer.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=True)


def replace_layers(model, old_layer=ReluLayer, new_layer=RedirectedReluLayer) -> None:
    """
    Replace all target layers with new layers.
    The most common use case is replacing activation layers with activation layers
    that can handle gradient flow issues.
    """

    for name, child in model._modules.items():
        if isinstance(child, old_layer):
            setattr(model, name, new_layer())
        elif child is not None:
            replace_layers(child, old_layer, new_layer)


class LocalResponseNormLayer(nn.Module):
    """
    Basic Hookable Local Response Norm layer.
    """

    def __init__(
        self,
        size: int = 9,
        alpha: float = 9.99999974738e-05,
        beta: float = 0.5,
        k: float = 1.0,
    ) -> None:
        super(LocalResponseNormLayer, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.local_response_norm(
            input, size=self.size, alpha=self.alpha, beta=self.beta, k=self.k
        )


class Conv2dSame(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = self.calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = self.calc_same_pad(iw, kw, self.stride[1], self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def pad_reflective_a4d(x: torch.Tensor, padding: List[int]) -> torch.Tensor:
    """
    Reflective padding for all 4 dimensions of an NCHW tensor
    """

    assert x.dim() == 4
    assert len(padding) == 8

    if x.size(0) == 2:
        assert sum(padding[6:]) == 0
    if x.size(1) == 2:
        assert sum(padding[4:6]) == 0

    # Pad batch
    if x.size(0) > 2:
        if padding[6] != 0:
            x = torch.cat([x, x.flip([0])[1 : (padding[6] + 1)]], dim=0)
        if padding[7] != 0:
            x = torch.cat([x.flip([0])[-(padding[7] + 1) : -1], x], dim=0)

    elif x.size(0) == 1:
        P = []
        if padding[6] != 0:
            P.append(
                torch.cat(
                    [x if (i + 1) % 2 == 0 else x.flip([0]) for i in range(padding[6])]
                )
            )
        P.append(x)
        if padding[7] != 0:
            P.append(
                torch.cat(
                    [x if (i + 1) % 2 == 0 else x.flip([0]) for i in range(padding[7])]
                )
            )
        x = torch.cat(P)

    # Pad channels
    if x.size(1) > 2:
        if padding[4] != 0:
            x = torch.cat([x, x.flip([1])[:, 1 : (padding[4] + 1)]], dim=1)
        if padding[5] != 0:
            x = torch.cat([x.flip([1])[:, -(padding[5] + 1) : -1], x], dim=1)

    elif x.size(1) == 1:
        P = []
        if padding[4] != 0:
            P.append(
                torch.cat(
                    [x if (i + 1) % 2 == 0 else x.flip([1]) for i in range(padding[4])],
                    dim=1,
                )
            )
        P.append(x)
        if padding[5] != 0:
            P.append(
                torch.cat(
                    [x if (i + 1) % 2 == 0 else x.flip([1]) for i in range(padding[5])],
                    dim=1,
                )
            )
        x = torch.cat(P, dim=1)

    # Pad height and width
    x = torch.nn.functional.pad(x, padding[:4], "reflect")
    return x


def collect_activations(
    model,
    targets: Union[nn.Module, List[nn.Module]],
    model_input: ModelInputType = torch.zeros(1, 3, 224, 224),
) -> ModuleOutputMapping:
    """
    Collect target activations for a model.
    """

    catch_activ = ActivationFetcher(model, targets)
    activ_out = catch_activ(model_input)
    return activ_out
