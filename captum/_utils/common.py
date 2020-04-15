#!/usr/bin/env python3
import typing
from enum import Enum
from inspect import signature
from typing import Any, Callable, List, Tuple, Union, cast, overload

import torch
from torch import Tensor, device
from torch.nn import Module

from .._utils.typing import Literal, TargetType


class ExpansionTypes(Enum):
    repeat = 1
    repeat_interleave = 2


def safe_div(
    denom: Tensor, quotient: Union[Tensor, float], default_value: Tensor
) -> Tensor:
    r"""
        A simple utility function to perform `denom / quotient`
        if the statement is undefined => result will be `default_value`
    """
    if isinstance(quotient, float):
        return denom / quotient if quotient != 0.0 else default_value

    # if quotient is a tensor
    return denom / torch.where(quotient != 0.0, quotient, default_value)


@typing.overload
def _is_tuple(inputs: Tensor) -> Literal[False]:
    ...


@typing.overload
def _is_tuple(inputs: Tuple[Tensor, ...]) -> Literal[True]:
    ...


def _is_tuple(inputs: Union[Tensor, Tuple[Tensor, ...]]) -> bool:
    return isinstance(inputs, tuple)


def _validate_target(num_samples: int, target: TargetType) -> None:
    if isinstance(target, list) or (
        isinstance(target, torch.Tensor) and torch.numel(target) > 1
    ):
        assert num_samples == len(target), (
            "The number of samples provied in the"
            "input {} does not match with the number of targets. {}".format(
                num_samples, len(target)
            )
        )


@overload
def _format_tensor_into_tuples(inputs: None) -> None:
    ...


@overload
def _format_tensor_into_tuples(
    inputs: Union[Tensor, Tuple[Tensor, ...]]
) -> Tuple[Tensor, ...]:
    ...


def _format_tensor_into_tuples(
    inputs: Union[None, Tensor, Tuple[Tensor, ...]]
) -> Union[None, Tuple[Tensor, ...]]:
    if inputs is None:
        return None
    if not isinstance(inputs, tuple):
        assert isinstance(
            inputs, torch.Tensor
        ), "`inputs` must have type " "torch.Tensor but {} found: ".format(type(inputs))
        inputs = (inputs,)
    return inputs


def _format_input(inputs: Union[Tensor, Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
    return _format_tensor_into_tuples(inputs)


@overload
def _format_additional_forward_args(additional_forward_args: None) -> None:
    ...


@overload
def _format_additional_forward_args(
    additional_forward_args: Union[Tensor, Tuple]
) -> Tuple:
    ...


@overload
def _format_additional_forward_args(additional_forward_args: Any) -> Union[None, Tuple]:
    ...


def _format_additional_forward_args(additional_forward_args: Any) -> Union[None, Tuple]:
    if additional_forward_args is not None and not isinstance(
        additional_forward_args, tuple
    ):
        additional_forward_args = (additional_forward_args,)
    return additional_forward_args


def _expand_additional_forward_args(
    additional_forward_args: Any,
    n_steps: int,
    expansion_type: ExpansionTypes = ExpansionTypes.repeat,
) -> Union[None, Tuple]:
    def _expand_tensor_forward_arg(
        additional_forward_arg: Tensor,
        n_steps: int,
        expansion_type: ExpansionTypes = ExpansionTypes.repeat,
    ) -> Tensor:
        if len(additional_forward_arg.size()) == 0:
            return additional_forward_arg
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([additional_forward_arg] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return additional_forward_arg.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    if additional_forward_args is None:
        return None

    return tuple(
        _expand_tensor_forward_arg(additional_forward_arg, n_steps, expansion_type)
        if isinstance(additional_forward_arg, torch.Tensor)
        else additional_forward_arg
        for additional_forward_arg in additional_forward_args
    )


def _expand_target(
    target: TargetType,
    n_steps: int,
    expansion_type: ExpansionTypes = ExpansionTypes.repeat,
) -> TargetType:
    if isinstance(target, list):
        if expansion_type == ExpansionTypes.repeat:
            return target * n_steps
        elif expansion_type == ExpansionTypes.repeat_interleave:
            expanded_target = []
            for i in target:
                expanded_target.extend([i] * n_steps)
            return cast(Union[List[Tuple[int, ...]], List[int]], expanded_target)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    elif isinstance(target, torch.Tensor) and torch.numel(target) > 1:
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([target] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return target.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    return target


def _run_forward(
    forward_func: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target: TargetType = None,
    additional_forward_args: Any = None,
) -> Tensor:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_input(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)

    output = forward_func(
        *(*inputs, *additional_forward_args)
        if additional_forward_args is not None
        else inputs
    )
    return _select_targets(output, target)


def _select_targets(output: Tensor, target: TargetType) -> Tensor:
    if target is None:
        return output

    num_examples = output.shape[0]
    dims = len(output.shape)
    device = output.device
    if isinstance(target, (int, tuple)):
        return _verify_select_column(output, target)
    elif isinstance(target, torch.Tensor):
        if torch.numel(target) == 1 and isinstance(target.item(), int):
            return _verify_select_column(output, cast(int, target.item()))
        elif len(target.shape) == 1 and torch.numel(target) == num_examples:
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(output, 1, target.reshape(len(output), 1))
        else:
            raise AssertionError(
                "Tensor target dimension %r is not valid. %r"
                % (target.shape, output.shape)
            )
    elif isinstance(target, list):
        assert len(target) == num_examples, "Target list length does not match output!"
        if isinstance(target[0], int):
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(
                output, 1, torch.tensor(target, device=device).reshape(len(output), 1)
            )
        elif isinstance(target[0], tuple):
            return torch.stack(
                [
                    output[(i,) + cast(Tuple, targ_elem)]
                    for i, targ_elem in enumerate(target)
                ]
            )
        else:
            raise AssertionError("Target element type in list is not valid.")
    else:
        raise AssertionError("Target type %r is not valid." % target)


def _verify_select_column(
    output: Tensor, target: Union[int, Tuple[int, ...]]
) -> Tensor:
    target = cast(Tuple[int, ...], (target,) if isinstance(target, int) else target)
    assert (
        len(target) <= len(output.shape) - 1
    ), "Cannot choose target column with output shape %r." % (output.shape,)
    return output[(slice(None), *target)]


def _extract_device(
    module: Module,
    hook_inputs: Union[None, Tensor, Tuple[Tensor, ...]],
    hook_outputs: Union[None, Tensor, Tuple[Tensor, ...]],
) -> device:
    params = list(module.parameters())
    if (
        (hook_inputs is None or len(hook_inputs) == 0)
        and (hook_outputs is None or len(hook_outputs) == 0)
        and len(params) == 0
    ):
        raise RuntimeError(
            """Unable to extract device information for the module
            {}. Both inputs and outputs to the forward hook and
            `module.parameters()` are empty.
            The reason that the inputs to the forward hook are empty
            could be due to the fact that the arguments to that
            module {} are all named and are passed as named
            variables to its forward function.
            """.format(
                module, module
            )
        )
    if hook_inputs is not None and len(hook_inputs) > 0:
        return hook_inputs[0].device
    if hook_outputs is not None and len(hook_outputs) > 0:
        return hook_outputs[0].device

    return params[0].device
