from typing import Optional

import torch
import torch.nn as nn

from captum.optim._param.image.transform import center_crop_shape
from captum.optim._utils.models import collect_activations
from captum.optim._utils.typing import TransformSize


def get_expanded_weights(
    model,
    target1: nn.Module,
    target2: nn.Module,
    crop_shape: Optional[TransformSize] = None,
    model_input: torch.Tensor = torch.zeros(1, 3, 224, 224),
) -> torch.Tensor:
    """
    Extract meaningful weight interactions from between neurons which aren’t
    literally adjacent in a neural network, or where the weights aren’t directly
    represented in a single weight tensor.
    Schubert, et al., "Visualizing Weights", Distill, 2020.
    See: https://distill.pub/2020/circuits/visualizing-weights/
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
