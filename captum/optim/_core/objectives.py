"""captum.optim.objectives."""

from contextlib import suppress
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from captum.optim._core.output_hook import AbortForwardException, ModuleOutputsHook
from captum.optim._param.image.images import InputParameterization, NaturalImage
from captum.optim._param.image.transform import RandomScale, RandomSpatialJitter
from captum.optim._utils.typing import (
    LossFunction,
    ModuleOutputMapping,
    Objective,
    Parameterized,
    SingleTargetLossFunction,
    StopCriteria,
)


class InputOptimization(Objective, Parameterized):
    """
    Core function that optimizes an input to maximize a target (aka objective).
    This is similar to gradient-based methods for adversarial examples, such
    as FGSM. The code for this was based on the implementation by the authors of Lucid.
    For more details, see the following:
        https://github.com/tensorflow/lucid
        https://distill.pub/2017/feature-visualization/
    """

    def __init__(
        self,
        model: nn.Module,
        input_param: Optional[InputParameterization],
        transform: Optional[nn.Module],
        target_modules: Iterable[nn.Module],
        loss_function: LossFunction,
        lr: float = 0.025,
    ):
        r"""
        Args:
            model (nn.Module):  The reference to PyTorch model instance.
            input_param (nn.Module, optional):  A module that generates an input,
                        consumed by the model.
            transform (nn.Module, optional):  A module that transforms or preprocesses
                        the input before being passed to the model.
            target_modules (iterable of nn.Module):  A list of targets, objectives that
                        are used to compute the loss function.
            loss_function (callable): The loss function to minimize during optimization
                        optimization.
            lr (float): The learning rate to use with the Adam optimizer.
        """
        self.model = model
        self.hooks = ModuleOutputsHook(target_modules)
        self.input_param = input_param or NaturalImage((224, 224))
        self.transform = transform or torch.nn.Sequential(
            RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05)), RandomSpatialJitter(16)
        )
        self.loss_function = loss_function
        self.lr = lr

    def loss(self) -> torch.Tensor:
        r"""Compute loss value for current iteration.
        Returns:
            *tensor* representing **loss**:
            - **loss** (*tensor*):
                        Size of the tensor corresponds to the targets passed.
        """
        image = self.input_param()._t[None, ...]

        if self.transform:
            image = self.transform(image)

        with suppress(AbortForwardException):
            _unreachable = self.model(image)  # noqa: F841

        # consume_outputs return the captured values and resets the hook's state
        module_outputs = self.hooks.consume_outputs()
        loss_value = self.loss_function(module_outputs)
        return loss_value

    def cleanup(self):
        r"""Garbage collection, mainly removing hooks."""
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

    def optimize(
        self,
        stop_criteria: Optional[StopCriteria] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        r"""Optimize input based on loss function and objectives.
        Args:
            stop_criteria (StopCriteria, optional):  A function that is called
                        every iteration and returns a bool that determines whether
                        to stop the optimization.
                        See captum.optim.typing.StopCriteria for details.
            optimizer (Optimizer, optional):  An torch.optim.Optimizer used to
                        optimize the input based on the loss function.
        Returns:
            *list* of *np.arrays* representing the **history**:
            - **history** (*list*):
                        A list of loss values per iteration.
                        Length of the list corresponds to the number of iterations
        """
        stop_criteria = stop_criteria or n_steps(1024)
        optimizer = optimizer or optim.Adam(self.parameters(), lr=self.lr)
        assert isinstance(optimizer, optim.Optimizer)

        history = []
        step = 0
        try:
            while stop_criteria(step, self, history, optimizer):
                optimizer.zero_grad()
                loss_value = self.loss()
                history.append(loss_value.cpu().detach().numpy())
                (-1 * loss_value.mean()).backward()
                optimizer.step()
                step += 1
        except (Exception, BaseException):
            self.cleanup()
        self.cleanup()
        return history


def n_steps(n: int) -> StopCriteria:
    """StopCriteria generator that uses number of steps as a stop criteria.
    Args:
        n (int):  Number of steps to run optimization.
    Returns:
        *StopCriteria* callable
    """
    pbar = tqdm(total=n, unit="step")

    def continue_while(step, obj, history, optim):
        if len(history) > 0:
            pbar.set_postfix({"Objective": f"{history[-1].mean():.1f}"}, refresh=False)
        if step < n:
            pbar.update()
            return True
        else:
            pbar.close()
            return False

    return continue_while


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
        model: nn.Module,
        target: nn.Module,
        loss_function: Callable[[torch.Tensor], torch.Tensor],
    ):
        super(SingleTargetObjective, self).__init__(model=model, targets=[target])
        self.loss_function = loss_function

    def loss(self, targets_to_values):
        assert len(self.targets) == 1
        target = self.targets[0]
        target_value = targets_to_values[target]
        loss_value = self.loss_function(target_value)
        self.history.append(loss_value.sum().cpu().detach().numpy().squeeze().item())
        return loss_value


class MultiTargetObjective(Objective):
    def __init__(
        self, objectives: List[Objective], weights: Optional[Iterable[float]] = None
    ):
        model = objectives[0].model
        assert all(o.model == model for o in objectives)
        targets = (target for objective in objectives for target in objective.targets)
        super(MultiTargetObjective, self).__init__(model=model, targets=targets)
        self.objectives = objectives
        self.weights = weights or len(objectives) * [1]

    def loss(self, targets_to_values):
        loss = (
            objective.loss_function(targets_to_values) for objective in self.objectives
        )
        weighted = (loss * weight for weight in self.weights)
        loss_value = sum(weighted)
        self.history.append(loss_value.cpu().detach().numpy().squeeze().item())
        return loss_value

    @property
    def histories(self) -> List[List[float]]:
        return [objective.history for objective in self.objectives]


# class ChannelObjective(SingleTargetObjective):
#     def __init__(self, channel: int, *args, **kwargs):
#         loss_function = lambda activation: activation[:, channel, :, :].mean()
#         super(ChannelObjective, self).__init__(
#             *args, loss_function=loss_function, **kwargs
#         )
