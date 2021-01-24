"""captum.optim.objectives."""

from contextlib import suppress
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " n_steps stop criteria with progress bar"
    )

from captum.optim._core.output_hook import AbortForwardException, ModuleOutputsHook
from captum.optim._param.image.images import InputParameterization, NaturalImage
from captum.optim._param.image.transform import RandomScale, RandomSpatialJitter
from captum.optim._utils.typing import (
    LossFunction,
    Objective,
    Parameterized,
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
        loss_function: LossFunction,
        input_param: Optional[InputParameterization] = None,
        transform: Optional[nn.Module] = None,
        lr: float = 0.025,
    ) -> None:
        r"""
        Args:
            model (nn.Module):  The reference to PyTorch model instance.
            input_param (nn.Module, optional):  A module that generates an input,
                        consumed by the model.
            transform (nn.Module, optional):  A module that transforms or preprocesses
                        the input before being passed to the model.
            loss_function (callable): The loss function to minimize during optimization
                        optimization.
            lr (float): The learning rate to use with the Adam optimizer.
        """
        self.model = model
        # Grab targets from loss_function
        if hasattr(loss_function.target, "__iter__"):
            self.hooks = ModuleOutputsHook(loss_function.target)
        else:
            self.hooks = ModuleOutputsHook([loss_function.target])
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
        input_t = (
            self.input_param()[None, ...]
            if self.input_param().dim() == 3
            else self.input_param()
        )

        if self.transform:
            input_t = self.transform(input_t)

        with suppress(AbortForwardException):
            _unreachable = self.model(input_t)  # noqa: F841

        # consume_outputs return the captured values and resets the hook's state
        module_outputs = self.hooks.consume_outputs()
        loss_value = self.loss_function(module_outputs)
        return loss_value

    def cleanup(self) -> None:
        r"""Garbage collection, mainly removing hooks."""
        self.hooks.remove_hooks()

    # Targets are managed by ModuleOutputHooks; we mainly just want a convenient setter
    @property
    def targets(self):
        return self.hooks.targets

    @targets.setter
    def targets(self, value) -> None:
        self.hooks.remove_hooks()
        self.hooks = ModuleOutputsHook(value)

    def parameters(self):
        return self.input_param.parameters()

    def optimize(
        self,
        stop_criteria: Optional[StopCriteria] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> List:
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
        except (Exception, BaseException) as e:
            self.cleanup()
            raise e
        self.cleanup()
        return history


def n_steps(n: int, show_progress: bool = True) -> StopCriteria:
    """StopCriteria generator that uses number of steps as a stop criteria.
    Args:
        n (int):  Number of steps to run optimization.
        show_progress (bool, optional):  Whether or not to show progress bar.
            Default: True
    Returns:
        *StopCriteria* callable
    """

    if show_progress:
        pbar = tqdm(total=n, unit=" step")

    def continue_while(step, obj, history, optim) -> bool:
        if len(history) > 0:
            if show_progress:
                pbar.set_postfix(
                    {"Objective": f"{history[-1].mean():.1f}"}, refresh=False
                )
        if step < n:
            if show_progress:
                pbar.update()
            return True
        else:
            if show_progress:
                pbar.close()
            return False

    return continue_while
