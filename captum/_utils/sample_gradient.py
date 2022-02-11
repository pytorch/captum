from collections import defaultdict
from enum import Enum
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from captum._utils.common import _format_tensor_into_tuples


def linear_param_grads(
    module: Module, activation: Tensor, gradient_out: Tensor, reset: bool = False
):
    if reset:
        module.weight.sample_grad = 0
        if module.bias is not None:
            module.bias.sample_grad = 0
    print(gradient_out.shape)
    print(activation.shape)

    module.weight.sample_grad += torch.einsum(
        "n...i,n...j->nij", gradient_out, activation
    )
    print(module.weight.sample_grad.shape)
    if module.bias is not None:
        module.bias.sample_grad += gradient_out


def conv2d_param_grads(
    module: Module, activation: Tensor, gradient_out: Tensor, reset: bool = False
):
    if reset:
        module.weight.sample_grad = 0
        if module.bias is not None:
            module.bias.sample_grad = 0

    batch_size = activation.shape[0]
    unfolded_act = torch.nn.functional.unfold(
        activation,
        module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    reshaped_grad = gradient_out.reshape(batch_size, -1, unfolded_act.shape[-1])
    grad1 = torch.einsum("ijk,ilk->ijl", reshaped_grad, unfolded_act)
    shape = [batch_size] + list(module.weight.shape)
    module.weight.sample_grad += grad1.reshape(shape)
    if module.bias is not None:
        module.bias.sample_grad += torch.sum(reshaped_grad, dim=2)


SUPPORTED_MODULES = {
    torch.nn.Conv2d: conv2d_param_grads,
    torch.nn.Linear: linear_param_grads,
}


class LossMode(Enum):
    SUM = 0
    MEAN = 1


class SampleGradientWrapper:
    def __init__(self, model):
        self.model = model
        self.hooks_added = False
        self.activation_dict = defaultdict(list)
        self.gradient_dict = defaultdict(list)
        self.forward_hooks = []
        self.backward_hooks = []

    def add_hooks(self):
        self.hooks_added = True
        self.model.apply(self._register_module_hooks)

    def _register_module_hooks(self, module: torch.nn.Module):
        if isinstance(module, tuple(SUPPORTED_MODULES.keys())):
            self.forward_hooks.append(
                module.register_forward_hook(self._forward_hook_fn)
            )
            self.backward_hooks.append(
                module.register_full_backward_hook(self._backward_hook_fn)
            )

    def _forward_hook_fn(
        self,
        module: Module,
        module_input: Union[Tensor, Tuple[Tensor, ...]],
        module_output: Union[Tensor, Tuple[Tensor, ...]],
    ):
        inp_tuple = _format_tensor_into_tuples(module_input)
        self.activation_dict[module].append(inp_tuple[0].clone().detach())

    def _backward_hook_fn(
        self,
        module: Module,
        grad_input: Union[Tensor, Tuple[Tensor, ...]],
        grad_output: Union[Tensor, Tuple[Tensor, ...]],
    ):
        grad_output_tuple = _format_tensor_into_tuples(grad_output)
        self.gradient_dict[module].append(grad_output_tuple[0].clone().detach())

    def remove_hooks(self):
        self.hooks_added = False

        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        self.forward_hooks = []
        self.backward_hooks = []

    def reset(self):
        self.activation_dict = defaultdict(list)
        self.gradient_dict = defaultdict(list)

    def compute_param_sample_gradients(self, loss_blob, loss_mode="mean"):
        assert (
            loss_mode.upper() in LossMode.__members__
        ), f"Provided loss mode {loss_mode} is not valid"
        mode = LossMode[loss_mode.upper()]

        self.model.zero_grad()
        loss_blob.backward(gradient=torch.ones_like(loss_blob))

        for module in self.activation_dict:
            sample_grad_fn = SUPPORTED_MODULES[type(module)]
            activations = self.activation_dict[module]
            gradients = self.gradient_dict[module]
            assert len(activations) == len(gradients), (
                "Number of saved activations do not match number of saved gradients."
                " This may occur if mmultiple forward passes are run without calling"
                " reset or computing param gradients."
            )
            for i, (act, grad) in enumerate(
                zip(activations, list(reversed(gradients)))
            ):
                mult = 1 if mode is LossMode.SUM else act.shape[0]
                sample_grad_fn(module, act, grad * mult, reset=(i == 0))
        self.reset()
