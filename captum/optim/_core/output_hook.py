from contextlib import suppress
from typing import Iterable, List, Union
from warnings import warn

import torch.nn as nn

from captum.optim._utils.typing import ModelInputType, ModuleOutputMapping


class AbortForwardException(Exception):
    pass


class ModuleReuseException(Exception):
    pass


# class SingleTargetHook:
#     def __init__(self, module: nn.Module):
#         self.saved_output = None
#         self.target_modules = [module]
#         self.remove_forward = module.register_forward_hook(self._forward_hook())

#     @property
#     def is_ready(self) -> bool:
#         return self.saved_output is not None

#     def _forward_hook(self):
#         def forward_hook(module, input, output):
#             assert self.module == module
#             self.saved_output = output
#             raise AbortForwardException("Forward hook called, output saved.")

#         return forward_hook

#     def __del__(self):
#         self.remove_forward()


class ModuleOutputsHook:
    def __init__(self, target_modules: Iterable[nn.Module]) -> None:
        # self.outputs: ModuleOutputMapping = dict.fromkeys(target_modules, None)
        self.outputs = dict.fromkeys(target_modules, None)
        self.hooks = [
            module.register_forward_hook(self._forward_hook())
            for module in target_modules
        ]

    def _reset_outputs(self) -> None:
        self.outputs = dict.fromkeys(self.outputs.keys(), None)

    @property
    def is_ready(self) -> bool:
        return all(value is not None for value in self.outputs.values())

    def _forward_hook(self):
        def forward_hook(module, input, output) -> None:
            assert module in self.outputs.keys()
            if self.outputs[module] is None:
                self.outputs[module] = output
            else:
                warn(
                    f"Hook attached to {module} was called multiple times. "
                    "As of 2019-11-22 please don't reuse nn.Modules in your models."
                )
            if self.is_ready:
                raise AbortForwardException("Forward hook called, all outputs saved.")

        return forward_hook

    def consume_outputs(self):  # -> ModuleOutputMapping:
        if not self.is_ready:
            warn(
                "Consume captured outputs, but not all requested target outputs "
                "have been captured yet!"
            )
        outputs = self.outputs
        self._reset_outputs()
        return outputs

    @property
    def targets(self):
        return self.outputs.keys()

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        # print(f"DEL HOOKS!: {list(self.outputs.keys())}")
        self.remove_hooks()


class ActivationFetcher:
    """
    Simple module for collecting activations from model targets.

    Args:
        model (nn.Module):  The reference to PyTorch model instance.
        targets (nn.module or list of nn.module):  The target layers to
            collect activations from.
    """

    def __init__(self, model, targets: Union[nn.Module, List[nn.Module]]) -> None:
        super(ActivationFetcher, self).__init__()
        self.model = model
        self.layers = ModuleOutputsHook(targets)

    def __call__(self, input_t: ModelInputType) -> ModuleOutputMapping:
        """
        Args:
            input_t (tensor or tuple of tensors, optional):  The input to use
                with the specified model.
        Returns:
            *dict*:  An dict containing the collected activations.
        """

        try:
            with suppress(AbortForwardException):
                self.model(input_t)
            activations = self.layers.consume_outputs()
        finally:
            self.layers.remove_hooks()
        return activations
