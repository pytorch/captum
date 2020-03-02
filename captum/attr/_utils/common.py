#!/usr/bin/env python3
import typing
from enum import Enum
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union, cast, overload

import torch
from torch import Tensor, device
from torch.nn import Module

from .approximation_methods import SUPPORTED_METHODS
from .typing import BaselineType, Literal, TargetType, TensorOrTupleOfTensorsGeneric

if TYPE_CHECKING:
    from .attribution import GradientAttribution


class ExpansionTypes(Enum):
    repeat = 1
    repeat_interleave = 2


@typing.overload
def _is_tuple(inputs: Tensor) -> Literal[False]:
    ...


@typing.overload
def _is_tuple(inputs: Tuple[Tensor, ...]) -> Literal[True]:
    ...


def _is_tuple(inputs: Union[Tensor, Tuple[Tensor, ...]]) -> bool:
    return isinstance(inputs, tuple)


def safe_div(denom: Tensor, quotient: float, default_value: Tensor) -> Tensor:
    r"""
        A simple utility function to perform `denom / quotient`
        if the statement is undefined => result will be `default_value`
    """
    return denom / quotient if quotient != 0.0 else default_value


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


def _validate_input(
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Union[Tensor, int, float], ...],
    n_steps: int = 50,
    method: str = "riemann_trapezoid",
    draw_baseline_from_distrib: bool = False,
) -> None:
    assert len(inputs) == len(baselines), (
        "Input and baseline must have the same "
        "dimensions, baseline has {} features whereas input has {}.".format(
            len(baselines), len(inputs)
        )
    )

    for input, baseline in zip(inputs, baselines):
        if draw_baseline_from_distrib:
            assert (
                isinstance(baseline, (int, float))
                or input.shape[1:] == baseline.shape[1:]
            ), (
                "The samples in input and baseline batches must have"
                " the same shape or the baseline corresponding to the"
                " input tensor must be a scalar."
                " Found baseline: {} and input: {} ".format(baseline, input)
            )
        else:
            assert (
                isinstance(baseline, (int, float))
                or input.shape == baseline.shape
                or baseline.shape[0] == 1
            ), (
                "Baseline can be provided as a tensor for just one input and"
                " broadcasted to the batch or input and baseline must have the"
                " same shape or the baseline corresponding to each input tensor"
                " must be a scalar. Found baseline: {} and input: {}".format(
                    baseline, input
                )
            )

    assert (
        n_steps >= 0
    ), "The number of steps must be a positive integer. " "Given: {}".format(n_steps)

    assert method in SUPPORTED_METHODS, (
        "Approximation method must be one for the following {}. "
        "Given {}".format(SUPPORTED_METHODS, method)
    )


def _validate_noise_tunnel_type(
    nt_type: str, supported_noise_tunnel_types: List[str]
) -> None:
    assert nt_type in supported_noise_tunnel_types, (
        "Noise types must be either `smoothgrad`, `smoothgrad_sq` or `vargrad`. "
        "Given {}".format(nt_type)
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


def _format_additional_forward_args(additional_forward_args: Any) -> Union[None, Tuple]:
    if additional_forward_args is not None and not isinstance(
        additional_forward_args, tuple
    ):
        additional_forward_args = (additional_forward_args,)
    return additional_forward_args


def _format_baseline(
    baselines: BaselineType, inputs: Tuple[Tensor, ...],
) -> Tuple[Union[Tensor, int, float], ...]:
    if baselines is None:
        return _zeros(inputs)

    if not isinstance(baselines, tuple):
        baselines = (baselines,)

    for baseline in baselines:
        assert isinstance(
            baseline, (torch.Tensor, int, float)
        ), "baseline input argument must be either a torch.Tensor or a number \
            however {} detected".format(
            type(baseline)
        )

    return baselines


@typing.overload
def _format_input_baseline(
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    baselines: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    ...


@typing.overload
def _format_input_baseline(
    inputs: Union[Tensor, Tuple[Tensor, ...]], baselines: BaselineType,
) -> Tuple[Tuple[Tensor, ...], Tuple[Union[Tensor, int, float], ...]]:
    ...


def _format_input_baseline(
    inputs: Union[Tensor, Tuple[Tensor, ...]], baselines: BaselineType,
) -> Tuple[Tuple[Tensor, ...], Tuple[Union[Tensor, int, float], ...]]:
    inputs = _format_input(inputs)
    baselines = _format_baseline(baselines, inputs)
    return inputs, baselines


# This function can potentially be merged with the `format_baseline` function
# however, since currently not all algorithms support baselines of type
# callable this will be kept in a separate function.
@typing.overload
def _format_callable_baseline(
    baselines: Union[
        None,
        Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        Tensor,
        Tuple[Tensor, ...],
    ],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Tensor, ...]:
    ...


@typing.overload
def _format_callable_baseline(
    baselines: Union[
        None,
        Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        Tensor,
        int,
        float,
        Tuple[Union[Tensor, int, float], ...],
    ],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Union[Tensor, int, float], ...]:
    ...


def _format_callable_baseline(
    baselines: Union[
        None,
        Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        Tensor,
        int,
        float,
        Tuple[Union[Tensor, int, float], ...],
    ],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Union[Tensor, int, float], ...]:
    if callable(baselines):
        # Note: this assumes that if baselines is a function and if it takes
        # arguments, then the first argument is the `inputs`.
        # This can be expanded in the future with better type checks
        baseline_parameters = signature(baselines).parameters
        if len(baseline_parameters) == 0:
            baselines = baselines()
        else:
            baselines = baselines(inputs)
    return _format_baseline(baselines, _format_input(inputs))


@typing.overload
def _format_attributions(
    is_inputs_tuple: Literal[True], attributions: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    ...


@typing.overload
def _format_attributions(
    is_inputs_tuple: Literal[False], attributions: Tuple[Tensor, ...]
) -> Tensor:
    ...


@typing.overload
def _format_attributions(
    is_inputs_tuple: bool, attributions: Tuple[Tensor, ...]
) -> Union[Tensor, Tuple[Tensor, ...]]:
    ...


def _format_attributions(
    is_inputs_tuple: bool, attributions: Tuple[Tensor, ...]
) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""
    In case input is a tensor and the attributions is returned in form of a
    tensor we take the first element of the attributions' tuple to match the
    same shape signatues of the inputs
    """
    assert isinstance(attributions, tuple), "Attributions must be in shape of a tuple"
    assert is_inputs_tuple or len(attributions) == 1, (
        "The input is a single tensor however the attributions aren't."
        "The number of attributed tensors is: {}".format(len(attributions))
    )
    return attributions if is_inputs_tuple else attributions[0]


def _format_and_verify_strides(
    strides: Union[None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]],
    inputs: Tuple[Tensor, ...],
) -> Tuple[Union[int, Tuple[int, ...]], ...]:
    # Formats strides, which are necessary for occlusion
    # Assumes inputs are already formatted (in tuple)
    if strides is None:
        strides = tuple(1 for input in inputs)
    if len(inputs) == 1 and not (isinstance(strides, tuple) and len(strides) == 1):
        strides = (strides,)  # type: ignore
    assert isinstance(strides, tuple) and len(strides) == len(
        inputs
    ), "Strides must be provided for each input tensor."
    for i in range(len(inputs)):
        assert isinstance(strides[i], int) or (
            isinstance(strides[i], tuple)
            and len(strides[i]) == len(inputs[i].shape) - 1  # type: ignore
        ), (
            "Stride for input index {} is {}, which is invalid for input with "
            "shape {}. It must be either an int or a tuple with length equal to "
            "len(input_shape) - 1."
        ).format(
            i, strides[i], inputs[i].shape
        )

    return strides


def _format_and_verify_sliding_window_shapes(
    sliding_window_shapes: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
    inputs: Tuple[Tensor, ...],
) -> Tuple[Tuple[int, ...], ...]:
    # Formats shapes of sliding windows, which is necessary for occlusion
    # Assumes inputs is already formatted (in tuple)
    if isinstance(sliding_window_shapes[0], int):
        sliding_window_shapes = (sliding_window_shapes,)  # type: ignore
    sliding_window_shapes: Tuple[Tuple[int, ...], ...]
    assert len(sliding_window_shapes) == len(
        inputs
    ), "Must provide sliding window dimensions for each input tensor."
    for i in range(len(inputs)):
        assert (
            isinstance(sliding_window_shapes[i], tuple)
            and len(sliding_window_shapes[i]) == len(inputs[i].shape) - 1
        ), (
            "Occlusion shape for input index {} is {} but should be a tuple with "
            "{} dimensions."
        ).format(
            i, sliding_window_shapes[i], len(inputs[i].shape) - 1
        )
    return sliding_window_shapes


@typing.overload
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Any,
    target: TargetType,
    is_inputs_tuple: Literal[False] = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    ...


@typing.overload
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Any,
    target: TargetType,
    is_inputs_tuple: Literal[True],
) -> Union[Tuple[Tensor, ...], Tuple[Tuple[Tensor, ...], Tensor]]:
    ...


# FIXME: GradientAttribution is provided as a string due to a circular import.
# This should be fixed when common is refactored into separate files.
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Any,
    target: TargetType,
    is_inputs_tuple: bool = False,
) -> Union[
    Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]
]:
    if return_convergence_delta:
        # computes convergence error
        delta = attr_algo.compute_convergence_delta(
            attributions,
            start_point,
            end_point,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        return _format_attributions(is_inputs_tuple, attributions), delta
    else:
        return _format_attributions(is_inputs_tuple, attributions)


def _zeros(inputs: Tuple[Tensor, ...]) -> Tuple[int, ...]:
    r"""
    Takes a tuple of tensors as input and returns a tuple that has the same
    length as `inputs` with each element as the integer 0.
    """
    return tuple(0 for input in inputs)


def _tensorize_baseline(
    inputs: Tuple[Tensor, ...], baselines: Tuple[Union[int, float, Tensor], ...]
) -> Tuple[Tensor, ...]:
    def _tensorize_single_baseline(baseline, input):
        if isinstance(baseline, (int, float)):
            return torch.full_like(input, baseline)
        if input.shape[0] > baseline.shape[0] and baseline.shape[0] == 1:
            return torch.cat([baseline] * input.shape[0])
        return baseline

    assert isinstance(inputs, tuple) and isinstance(baselines, tuple), (
        "inputs and baselines must"
        "have tuple type but found baselines: {} and inputs: {}".format(
            type(baselines), type(inputs)
        )
    )
    return tuple(
        _tensorize_single_baseline(baseline, input)
        for baseline, input in zip(baselines, inputs)
    )


def _reshape_and_sum(
    tensor_input: Tensor, num_steps: int, num_examples: int, layer_size: Tuple[int, ...]
) -> Tensor:
    # Used for attribution methods which perform integration
    # Sums across integration steps by reshaping tensor to
    # (num_steps, num_examples, (layer_size)) and summing over
    # dimension 0. Returns a tensor of size (num_examples, (layer_size))
    return torch.sum(
        tensor_input.reshape((num_steps, num_examples) + layer_size), dim=0
    )


def _verify_select_column(
    output: Tensor, target: Union[int, Tuple[int, ...]]
) -> Tensor:
    target = cast(Tuple[int, ...], (target,) if isinstance(target, int) else target)
    assert (
        len(target) <= len(output.shape) - 1
    ), "Cannot choose target column with output shape %r." % (output.shape,)
    return output[(slice(None), *target)]


def _select_targets(output: Tensor, target: TargetType) -> Tensor:
    if target is None:
        return output

    num_examples = output.shape[0]
    dims = len(output.shape)
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
            return torch.gather(output, 1, torch.tensor(target).reshape(len(output), 1))
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


def _call_custom_attribution_func(
    custom_attribution_func: Callable[..., Tuple[Tensor, ...]],
    multipliers: Tuple[Tensor, ...],
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    assert callable(custom_attribution_func), (
        "`custom_attribution_func`"
        " must be a callable function but {} provided".format(
            type(custom_attribution_func)
        )
    )
    custom_attr_func_params = signature(custom_attribution_func).parameters

    if len(custom_attr_func_params) == 1:
        return custom_attribution_func(multipliers)
    elif len(custom_attr_func_params) == 2:
        return custom_attribution_func(multipliers, inputs)
    elif len(custom_attr_func_params) == 3:
        return custom_attribution_func(multipliers, inputs, baselines)
    else:
        raise AssertionError(
            "`custom_attribution_func` must take at least one and at most 3 arguments."
        )


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


def _find_output_mode_and_verify(
    initial_eval: Union[int, float, Tensor],
    num_examples: int,
    perturbations_per_eval: int,
    feature_mask: Union[None, TensorOrTupleOfTensorsGeneric],
) -> bool:
    """
    This method identifies whether the model outputs a single output for a batch
    (agg_output_mode = True) or whether it outputs a single output per example
    (agg_output_mode = False) and returns agg_output_mode. The method also
    verifies that perturbations_per_eval is 1 in the case that agg_output_mode is True
    and also verifies that the first dimension of each feature mask if the model
    returns a single output for a batch.
    """
    if isinstance(initial_eval, (int, float)) or (
        isinstance(initial_eval, torch.Tensor)
        and (
            len(initial_eval.shape) == 0
            or (num_examples > 1 and initial_eval.numel() == 1)
        )
    ):
        agg_output_mode = True
        assert (
            perturbations_per_eval == 1
        ), "Cannot have perturbations_per_eval > 1 when function returns scalar."
        if feature_mask is not None:
            for single_mask in feature_mask:
                assert single_mask.shape[0] == 1, (
                    "Cannot provide different masks for each example when function "
                    "returns a scalar."
                )
    else:
        agg_output_mode = False
        assert (
            isinstance(initial_eval, torch.Tensor) and initial_eval[0].numel() == 1
        ), "Target should identify a single element in the model output."
    return agg_output_mode
