from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from captum.optim._param.image.transform import center_crop_shape
from captum.optim._utils.models import collect_activations
from captum.optim._utils.typing import ModelInputType, PoolParam, TransformSize


def get_expanded_weights(
    model,
    target1: nn.Module,
    target2: nn.Module,
    crop_shape: Optional[Union[Tuple[int, int], TransformSize]] = None,
    model_input: ModelInputType = torch.zeros(1, 3, 224, 224),
) -> torch.Tensor:
    """
    Extract meaningful weight interactions from between neurons which aren’t
    literally adjacent in a neural network, or where the weights aren’t directly
    represented in a single weight tensor.
    Schubert, et al., "Visualizing Weights", Distill, 2020.
    See: https://distill.pub/2020/circuits/visualizing-weights/

    Args:
        model (nn.Module):  The reference to PyTorch model instance.
        target1 (nn.module):  The starting target layer. Must be below the layer
            specified for target2.
        target2 (nn.Module):  The end target layer. Must be above the layer
            specified for target1.
        crop_shape (int or tuple of ints, optional):  Specify the output weight
            size to enter crop away padding.
        model_input (tensor or tuple of tensors, optional):  The input to use
            with the specified model.
    Returns:
        *tensor*:  A tensor containing the expanded weights in the form of:
            (target2 output channels, target1 output channels, y, x)
    """

    activations = collect_activations(model, [target1, target2], model_input)
    activ1 = activations[target1]
    activ2 = activations[target2]

    if activ2.dim() == 4:
        t_offset_h, t_offset_w = (activ2.size(2) - 1) // 2, (activ2.size(3) - 1) // 2
        t_center = activ2[:, :, t_offset_h, t_offset_w]
    elif activ2.dim() == 2:
        t_center = activ2

    A = []
    for i in range(activ2.size(1)):
        x = torch.autograd.grad(
            outputs=t_center[:, i],
            inputs=[activ1],
            grad_outputs=torch.ones_like(t_center[:, i]),
            retain_graph=True,
        )[0]
        A.append(x.squeeze(0))
    expanded_weights = torch.stack(A, 0)

    if crop_shape is not None:
        expanded_weights = center_crop_shape(expanded_weights, crop_shape)
    return expanded_weights


def max2avg_pool2d(model, value: Optional[Any] = float("-inf")) -> None:
    """
    Replace all nonlinear MaxPool2d layers with their linear AvgPool2d equivalents.
    This allows us to ignore nonlinear values when calculating expanded weights.

    Args:
        model (nn.Module): A PyTorch model instance.
        value (Any): Used to return any padding that's meant to be ignored by
            pooling layers back to zero.
    """

    class AvgPool2dInf(torch.nn.Module):
        def __init__(
            self,
            kernel_size: PoolParam = 2,
            stride: Optional[PoolParam] = 2,
            padding: PoolParam = 0,
            ceil_mode: bool = False,
            value: Optional[Any] = None,
        ) -> None:
            super().__init__()
            self.avgpool = torch.nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
            )
            self.value = value

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.avgpool(x)
            if self.value is not None:
                x[x == self.value] = 0.0
            return x

    for name, child in model._modules.items():
        if isinstance(child, torch.nn.MaxPool2d):
            new_layer = AvgPool2dInf(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                ceil_mode=child.ceil_mode,
                value=value,
            )
            setattr(model, name, new_layer)
        elif child is not None:
            max2avg_pool2d(child)


def ignore_layer(model, layer) -> None:
    """
    Replace target layers with layers that do nothing.
    This is useful for removing the nonlinear ReLU
    layers when creating expanded weights.

    Args:
        model (nn.Module): A PyTorch model instance.
        layer (nn.Module): A layer class type.
    """

    class IgnoreLayer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    for name, child in model._modules.items():
        if isinstance(child, layer):
            new_layer = IgnoreLayer()
            setattr(model, name, new_layer)
        elif child is not None:
            ignore_layer(child, layer)
