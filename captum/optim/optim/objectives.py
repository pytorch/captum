from contextlib import suppress
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from clarity.pytorch import Parameterized, Objective, ModuleOutputMapping
from clarity.pytorch.param import ImageParameterization, NaturalImage, RandomAffine

from .output_hook import AbortForwardException, ModuleOutputsHook

LossFunction = Callable[[ModuleOutputMapping], torch.Tensor]
SingleTargetLossFunction = Callable[[torch.Tensor], torch.Tensor]


class InputOptimization(Objective, Parameterized):
    net: nn.Module
    input_param: ImageParameterization
    input_transformation: nn.Module

    def __init__(
        self,
        net: nn.Module,
        input_param: Optional[nn.Module],
        transform: Optional[nn.Module],
        targets: Iterable[nn.Module],
        loss_function: LossFunction,
    ):
        self.net = net
        self.hooks = ModuleOutputsHook(targets)
        self.input_param = input_param or NaturalImage((224, 224))
        self.transform = transform or RandomAffine(scale=True, translate=True)
        self.loss_function = loss_function

    def loss(self) -> torch.Tensor:
        image = self.input_param()[None, ...]

        if self.transform:
            image = self.transform(image)

        with suppress(AbortForwardException):
            _unreachable = self.net(image)

        # consume_ouputs return the captured values and resets the hook's state
        module_outputs = self.hooks.consume_outputs()
        loss_value = self.loss_function(module_outputs)
        return loss_value

    def cleanup(self):
        self.hooks.remove_hooks()

    # Targets are managed by ModuleOutputHooks; we mainly just want a convenient setter
    @property
    def targets(self):
        return self.hooks.targets

    @targets.setter
    def targets(self, value):
        self.hooks.remove_hooks()
        self.hooks = ModuleOutputsHook(value)

    def parameters(self):
        return self.input_param.parameters()


def channel_activation(target: nn.Module, channel_index: int) -> LossFunction:
    # ensure channel_index will be valid
    assert channel_index < target.out_channels

    def loss_function(targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[target]
        assert activations is not None
        assert len(activations.shape) == 4  # assume NCHW
        return activations[:, channel_index, ...]

    return loss_function


def neuron_activation(
    target: nn.Module, channel_index: int, x: int = None, y: int = None
) -> LossFunction:
    # ensure channel_index will be valid
    assert channel_index < target.out_channels

    def loss_function(targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[target]
        assert activations is not None
        assert len(activations.shape) == 4  # assume NCHW
        _, _, H, W = activations.shape

        if x is None:
            _x = W // 2
        else:
            assert x < W
            _x = x

        if y is None:
            _y = H // 2
        else:
            assert y < W
            _y = y

        return activations[:, channel_index, _x, _y]

    return loss_function


def single_target_objective(
    target: nn.Module, loss_function: SingleTargetLossFunction
) -> LossFunction:
    def inner(targets_to_values: ModuleOutputMapping):
        value = targets_to_values[target]
        return loss_function(value)

    return inner


class SingleTargetObjective(Objective):
    def __init__(
        self,
        net: nn.Module,
        target: nn.Module,
        loss_function: Callable[[torch.Tensor], torch.Tensor],
    ):
        super(SingleTargetObjective, self).__init__(net=net, targets=[target])
        self.loss_function = loss_function

    def loss(self, targets_to_values):
        assert len(self.targets) == 1
        target = self.targets[0]
        target_value = targets_to_values[target]
        loss_value = self.loss_function(target_value)
        self.history.append(loss_value.sum().cpu().detach().numpy().squeeze().item())
        return loss_value


# class MultiObjective(Objective):
#     def __init__(
#         self, objectives: List[Objective], weights: Optional[Iterable[float]] = None
#     ):
#         net = objectives[0].net
#         assert all(o.net == net for o in objectives)
#         targets = (target for objective in objectives for target in objective.targets)
#         super(MultiObjective, self).__init__(net=net, targets=targets)
#         self.objectives = objectives
#         self.weights = weights or len(objectives) * [1]

#     def loss(self, targets_to_values):
#         losses = (
#             objective.loss_function(targets_to_values) for objective in self.objectives
#         )
#         weighted = (loss * weight for weight in self.weights)
#         loss_value = sum(weighted)
#         self.history.append(loss_value.cpu().detach().numpy().squeeze().item())
#         return loss_value

#     @property
#     def histories(self) -> List[List[float]]:
#         return [objective.history for objective in self.objectives]


# class ChannelObjective(SingleTargetObjective):
#     def __init__(self, channel: int, *args, **kwargs):
#         loss_function = lambda activation: activation[:, channel, :, :].mean()
#         super(ChannelObjective, self).__init__(
#             *args, loss_function=loss_function, **kwargs
#         )

