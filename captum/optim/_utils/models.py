from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    See https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
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
