from typing import List, Optional

import torch
import torch.nn as nn

from captum.optim._param.image.transform import center_crop_shape
from captum.optim._utils.models import collect_activations
from captum.optim._utils.typing import ModelInputType, TransformSize


def get_expanded_weights(
    model,
    target1: nn.Module,
    target2: nn.Module,
    crop_shape: Optional[TransformSize] = None,
    model_input: ModelInputType = torch.zeros(1, 3, 224, 224),
) -> torch.Tensor:
    """
    Extract meaningful weight interactions from between neurons which aren’t
    literally adjacent in a neural network, or where the weights aren’t directly
    represented in a single weight tensor.
    Schubert, et al., "Visualizing Weights", Distill, 2020.
    See: https://distill.pub/2020/circuits/visualizing-weights/

    Args:
        model:  PyTorch model instance.
        target1 (nn.module):  The starting target layer. Must be below the layer
            specified for target2.
        target2 (nn.module):  The end target layer. Must be above the layer
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
    exapnded_weights = torch.stack(A, 0)

    if crop_shape is not None:
        exapnded_weights = center_crop_shape(exapnded_weights, crop_shape)
    return exapnded_weights


def tensor_heatmap(
    tensor: torch.Tensor,
    colors: List[str] = ["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"],
) -> torch.Tensor:
    """
    Create a color heatmap of an input weight tensor.
    By default red represents excitatory values,
    blue represents inhibitory values, and white represents
    no excitation or inhibition.
    """

    assert tensor.dim() == 2
    assert len(colors) == 5

    def get_color(x: str) -> torch.Tensor:
        def hex2base10(x: str) -> float:
            return int(x, 16) / 255.0

        return torch.tensor(
            [hex2base10(x[0:2]), hex2base10(x[2:4]), hex2base10(x[4:6])]
        )

    def color_scale(x: torch.Tensor) -> torch.Tensor:
        if x < 0:
            x = -x
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2]) + x * get_color(colors[1])
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[1]) + x * get_color(colors[0])
        else:
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2]) + x * get_color(colors[3])
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[3]) + x * get_color(colors[4])

    return torch.stack(
        [torch.stack([color_scale(x) for x in t]) for t in tensor]
    ).permute(2, 0, 1)
