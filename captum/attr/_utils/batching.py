#!/usr/bin/env python3
import typing
import warnings
from typing import Any, Callable, Iterator, Tuple, Union

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _reduce_list,
)
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
    TupleOrTensorOrBoolGeneric,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from torch import Tensor


def _batch_attribution(
    attr_method,
    num_examples,
    internal_batch_size,
    n_steps,
    include_endpoint=False,
    **kwargs,
):
    """
    This method applies internal batching to given attribution method, dividing
    the total steps into batches and running each independently and sequentially,
    adding each result to compute the total attribution.

    Step sizes and alphas are spliced for each batch and passed explicitly for each
    call to _attribute.

    kwargs include all argument necessary to pass to each attribute call, except
    for n_steps, which is computed based on the number of steps for the batch.

    include_endpoint ensures that one step overlaps between each batch, which
    is necessary for some methods, particularly LayerConductance.
    """
    if internal_batch_size < num_examples:
        warnings.warn(
            "Internal batch size cannot be less than the number of input examples. "
            "Defaulting to internal batch size of %d equal to the number of examples."
            % num_examples
        )
    # Number of steps for each batch
    step_count = max(1, internal_batch_size // num_examples)
    if include_endpoint:
        if step_count < 2:
            step_count = 2
            warnings.warn(
                "This method computes finite differences between evaluations at "
                "consecutive steps, so internal batch size must be at least twice "
                "the number of examples. Defaulting to internal batch size of %d"
                " equal to twice the number of examples." % (2 * num_examples)
            )

    total_attr = None
    cumulative_steps = 0
    step_sizes_func, alphas_func = approximation_parameters(kwargs["method"])
    full_step_sizes = step_sizes_func(n_steps)
    full_alphas = alphas_func(n_steps)

    while cumulative_steps < n_steps:
        start_step = cumulative_steps
        end_step = min(start_step + step_count, n_steps)
        batch_steps = end_step - start_step

        if include_endpoint:
            batch_steps -= 1

        step_sizes = full_step_sizes[start_step:end_step]
        alphas = full_alphas[start_step:end_step]
        current_attr = attr_method._attribute(
            **kwargs, n_steps=batch_steps, step_sizes_and_alphas=(step_sizes, alphas)
        )

        if total_attr is None:
            total_attr = current_attr
        else:
            if isinstance(total_attr, Tensor):
                total_attr = total_attr + current_attr.detach()
            else:
                total_attr = tuple(
                    current.detach() + prev_total
                    for current, prev_total in zip(current_attr, total_attr)
                )
        if include_endpoint and end_step < n_steps:
            cumulative_steps = end_step - 1
        else:
            cumulative_steps = end_step
    return total_attr


@typing.overload
def _tuple_splice_range(inputs: None, start: int, end: int) -> None:
    ...


@typing.overload
def _tuple_splice_range(inputs: Tuple, start: int, end: int) -> Tuple:
    ...


def _tuple_splice_range(
    inputs: Union[None, Tuple], start: int, end: int
) -> Union[None, Tuple]:
    """
    Splices each tensor element of given tuple (inputs) from range start
    (inclusive) to end (non-inclusive) on its first dimension. If element
    is not a Tensor, it is left unchanged. It is assumed that all tensor elements
    have the same first dimension (corresponding to number of examples).
    The returned value is a tuple with the same length as inputs, with Tensors
    spliced appropriately.
    """
    assert start < end, "Start point must precede end point for batch splicing."
    if inputs is None:
        return None
    return tuple(
        inp[start:end] if isinstance(inp, torch.Tensor) else inp for inp in inputs
    )


def _batched_generator(
    inputs: TensorOrTupleOfTensorsGeneric,
    additional_forward_args: Any = None,
    target_ind: TargetType = None,
    internal_batch_size: Union[None, int] = None,
) -> Iterator[Tuple[Tuple[Tensor, ...], Any, TargetType]]:
    """
    Returns a generator which returns corresponding chunks of size internal_batch_size
    for both inputs and additional_forward_args. If batch size is None,
    generator only includes original inputs and additional args.
    """
    assert internal_batch_size is None or (
        isinstance(internal_batch_size, int) and internal_batch_size > 0
    ), "Batch size must be greater than 0."
    inputs = _format_tensor_into_tuples(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    num_examples = inputs[0].shape[0]
    # TODO Reconsider this check if _batched_generator is used for non gradient-based
    # attribution algorithms
    if not (inputs[0] * 1).requires_grad:
        warnings.warn(
            """It looks like that the attribution for a gradient-based method is
            computed in a `torch.no_grad` block or perhaps the inputs have no
            requires_grad."""
        )
    if internal_batch_size is None:
        yield inputs, additional_forward_args, target_ind
    else:
        for current_total in range(0, num_examples, internal_batch_size):
            with torch.autograd.set_grad_enabled(True):
                inputs_splice = _tuple_splice_range(
                    inputs, current_total, current_total + internal_batch_size
                )
            yield inputs_splice, _tuple_splice_range(
                additional_forward_args,
                current_total,
                current_total + internal_batch_size,
            ), target_ind[
                current_total : current_total + internal_batch_size
            ] if isinstance(
                target_ind, list
            ) or (
                isinstance(target_ind, torch.Tensor) and target_ind.numel() > 1
            ) else target_ind


def _batched_operator(
    operator: Callable[..., TupleOrTensorOrBoolGeneric],
    inputs: TensorOrTupleOfTensorsGeneric,
    additional_forward_args: Any = None,
    target_ind: TargetType = None,
    internal_batch_size: Union[None, int] = None,
    **kwargs: Any,
) -> TupleOrTensorOrBoolGeneric:
    """
    Batches the operation of the given operator, applying the given batch size
    to inputs and additional forward arguments, and returning the concatenation
    of the results of each batch.
    """
    all_outputs = [
        operator(
            inputs=input,
            additional_forward_args=additional,
            target_ind=target,
            **kwargs,
        )
        for input, additional, target in _batched_generator(
            inputs, additional_forward_args, target_ind, internal_batch_size
        )
    ]
    return _reduce_list(all_outputs)


def _select_example(curr_arg: Any, index: int, bsz: int) -> Any:
    if curr_arg is None:
        return None
    is_tuple = isinstance(curr_arg, tuple)
    if not is_tuple:
        curr_arg = (curr_arg,)
    selected_arg = []
    for i in range(len(curr_arg)):
        if isinstance(curr_arg[i], (Tensor, list)) and len(curr_arg[i]) == bsz:
            selected_arg.append(curr_arg[i][index : index + 1])
        else:
            selected_arg.append(curr_arg[i])
    return _format_output(is_tuple, tuple(selected_arg))


def _batch_example_iterator(bsz: int, *args) -> Iterator:
    """
    Batches the provided argument.
    """
    for i in range(bsz):
        curr_args = [_select_example(args[j], i, bsz) for j in range(len(args))]
        yield tuple(curr_args)
