#!/usr/bin/env python3

from .common import _reduce_list
from .common import format_input, _format_additional_forward_args, tuple_splice_range


def _batched_generator(inputs, additional_forward_args=None, batch_size=None):
    inputs = format_input(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    num_examples = inputs[0].shape[0]
    if batch_size is None:
        yield inputs, additional_forward_args
    else:
        current_total = 0
        while current_total < num_examples:
            yield tuple_splice_range(
                inputs, current_total, current_total + batch_size
            ), tuple_splice_range(
                additional_forward_args, current_total, current_total + batch_size
            )
            current_total += batch_size


def _batched_operator(
    operator, inputs, additional_forward_args=None, batch_size=None, **kwargs
):
    all_outputs = []
    for input, additional in _batched_generator(
        inputs, additional_forward_args, batch_size
    ):
        all_outputs.append(
            operator(inputs=input, additional_forward_args=additional, **kwargs)
        )
    return _reduce_list(all_outputs)
