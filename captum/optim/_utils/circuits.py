from typing import List, Union

import torch
import torch.nn as nn

from captum.optim._utils.typing import ModuleOutputMapping
from captum.optim._utils.circuits import ActivationCatcher


def get_expanded_weights(model, target1: nn.Module, target2: nn.Module) -> torch.Tensor:
    """
    Extract meaningful weight interactions from between neurons which aren’t
    literally adjacent in a neural network, or where the weights aren’t directly
    represented in a single weight tensor.

    Schubert, et al., "Visualizing Weights", Distill, 2020.
    See: https://distill.pub/2020/circuits-visualizing-weights/
    """

    def get_activations(
        model, targets: Union[nn.Module, List[nn.Module]]
    ) -> ModuleOutputMapping:
        catch_activ = ActivationCatcher(targets)
        activ_out = catch_activ(model, torch.zeros(1, 3, 224, 224))
        return activ_out

    activations = get_activations(model, [target1, target2])
    activ1 = activations[target1]
    activ2 = activations[target2]

    t_offset = (activ2.size(2) - 1) // 2
    t_center = activ2[0, :, t_offset, t_offset]

    A = []
    for i in range(activ2.size(1)):
        x = torch.autograd.grad(
            outputs=t_center[i],
            inputs=[activ1],
            grad_outputs=torch.ones_like(t_center[i]),
            retain_graph=True,
        )[0]
        A.append(x)

    return torch.stack(A)
