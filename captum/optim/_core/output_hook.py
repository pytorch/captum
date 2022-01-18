import warnings
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn

from captum.optim._utils.typing import ModuleOutputMapping, TupleOfTensorsOrTensorType


class ModuleOutputsHook:
    def __init__(self, target_modules: Iterable[nn.Module]) -> None:
        """
        Args:

            target_modules (Iterable of nn.Module): A list of nn.Module targets.
        """
        self.outputs: ModuleOutputMapping = dict.fromkeys(target_modules, None)
        self.hooks = [
            module.register_forward_hook(self._forward_hook())
            for module in target_modules
        ]

    def _reset_outputs(self) -> None:
        """
        Delete captured activations.
        """
        self.outputs = dict.fromkeys(self.outputs.keys(), None)

    @property
    def is_ready(self) -> bool:
        return all(value is not None for value in self.outputs.values())

    def _forward_hook(self) -> Callable:
        """
        Return the module_outputs_forward_hook forward hook function.

        Returns:
            forward_hook (Callable): The module_outputs_forward_hook function.
        """

        def module_outputs_forward_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            assert module in self.outputs.keys()
            if self.outputs[module] is None:
                self.outputs[module] = output
            else:
                warn(
                    f"Hook attached to {module} was called multiple times. "
                    "As of 2019-11-22 please don't reuse nn.Modules in your models."
                )
            if self.is_ready:
                warn(
                    "No outputs found from models. This can be ignored if you are "
                    "optimizing on inputs only, without models. Otherwise, check "
                    "that you are passing model layers in your losses."
                )

        return module_outputs_forward_hook

    def consume_outputs(self) -> ModuleOutputMapping:
        """
        Collect target activations and return them.

        Returns:
            outputs (ModuleOutputMapping): The captured outputs.
        """
        if not self.is_ready:
            warn(
                "Consume captured outputs, but not all requested target outputs "
                "have been captured yet!"
            )
        outputs = self.outputs
        self._reset_outputs()
        return outputs

    @property
    def targets(self) -> Iterable[nn.Module]:
        return self.outputs.keys()

    def remove_hooks(self) -> None:
        """
        Remove hooks.
        """
        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        """
        Ensure that using 'del' properly deletes hooks.
        """
        self.remove_hooks()


class ActivationFetcher:
    """
    Simple module for collecting activations from model targets.
    """

    def __init__(self, model: nn.Module, targets: Iterable[nn.Module]) -> None:
        """
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            targets (nn.Module or list of nn.Module):  The target layers to
                collect activations from.
        """
        super(ActivationFetcher, self).__init__()
        self.model = model
        self.layers = ModuleOutputsHook(targets)

    def __call__(self, input_t: TupleOfTensorsOrTensorType) -> ModuleOutputMapping:
        """
        Args:

            input_t (tensor or tuple of tensors, optional):  The input to use
                with the specified model.

        Returns:
            activations_dict: An dict containing the collected activations. The keys
                for the returned dictionary are the target layers.
        """

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model(input_t)
            activations_dict = self.layers.consume_outputs()
        finally:
            self.layers.remove_hooks()
        return activations_dict


def _remove_all_forward_hooks(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.

    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safely identify and remove
    the target forward hooks.

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        hook_fn_name (str, optional): Optionally only remove specific forward hooks
            based on their function's __name__ attribute.
            Default: None
    """

    if hook_fn_name is None:
        warn("Removing all active hooks will break some PyTorch modules & systems.")

    def _remove_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(module, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                if name is not None:
                    dict_items = list(m._forward_hooks.items())
                    m._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != name]
                    )
                else:
                    m._forward_hooks: Dict[int, Callable] = OrderedDict()

    def _remove_child_hooks(
        target_module: torch.nn.Module, hook_name: Optional[str] = None
    ) -> None:
        for name, child in target_module._modules.items():
            if child is not None:
                _remove_hooks(child, hook_name)
                _remove_child_hooks(child, hook_name)

    # Remove hooks from target submodules
    _remove_child_hooks(module, hook_fn_name)

    # Remove hooks from the target module
    _remove_hooks(module, hook_fn_name)


def cleanup_module_hooks(modules: Union[nn.Module, List[nn.Module]]) -> None:
    """
    Remove any InputOptimization hooks from the specified modules. This may be useful
    in the event that something goes wrong in between creating the InputOptimization
    instance and running the optimization function, or if InputOptimization fails
    without properly removing it's hooks.

    Warning: This function will remove all the hooks placed by InputOptimization
    instances on the target modules, and thus can interfere with using multiple
    InputOptimization instances.

    Args:

        modules (nn.Module or list of nn.Module): Any module instances that contain
            hooks created by InputOptimization, for which the removal of the hooks is
            required.
    """
    if not hasattr(modules, "__iter__"):
        modules = [modules]
    # Captum ModuleOutputsHook uses "module_outputs_forward_hook" hook functions
    [
        _remove_all_forward_hooks(module, "module_outputs_forward_hook")
        for module in modules
    ]


__all__ = ["cleanup_module_hooks"]
