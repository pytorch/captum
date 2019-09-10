#!/usr/bin/env python3
import torch

from .common import format_input, _format_additional_forward_args


def _tuple_splice_range(inputs, start, end):
    """
    Splices each tensor element of given tuple (inputs) from range start
    (inclusive) to end (non-inclusive) on its first dimension. If element
    is not a Tensor, it is left unchanged. The returned value is a tuple
    with the same length as inputs, with Tensors spliced appropriately.
    """
    if inputs is None:
        return None
    return tuple(
        inp[start:end] if isinstance(inp, torch.Tensor) else inp for inp in inputs
    )


def _reduce_list(val_list, red_func=torch.cat):
    """
    Applies reduction function to given list. If each element in the list is
    a Tensor, applies reduction function to all elements of the list, and returns
    the output Tensor / value. If each element is a tuple, applies reduction
    function to corresponding elements of each tuple in the list, and returns
    tuple of reduction function outputs with length matching the length of tuple
    val_list[0].
    """
    if isinstance(val_list[0], torch.Tensor):
        return red_func(val_list)
    assert isinstance(val_list[0], tuple), "Elements to be reduced can only be"
    "either Tensors or tuples containing Tensors."
    final_out = []
    for i in range(len(val_list[0])):
        final_out.append(_reduce_list([val_elem[i] for val_elem in val_list], red_func))
    return tuple(final_out)


def _sort_key_list(keys, device_ids=None):
    """
    Sorts list of torch devices (keys) by given index list, device_ids. If keys
    contains only one device, then the list is returned unchanged. If keys
    contains a device for which the id is not contained in device_ids, then
    an error is returned. This method is used to identify the order of DataParallel
    batched devices, given the device ID ordering.
    """
    if len(keys) == 1:
        return keys
    id_dict = {}
    assert device_ids is not None, "Device IDs must be provided with multiple devices."
    for key in keys:
        if key.index in id_dict:
            raise AssertionError("Duplicate CUDA Device ID identified in device list.")
        id_dict[key.index] = key

    out_list = []
    for dev_id in device_ids:
        if dev_id in id_dict:
            out_list.append(id_dict[dev_id])

    assert len(out_list) == len(keys), "Given Device ID List does not match"
    "devices with computed tensors."

    return out_list


def _batched_generator(inputs, additional_forward_args=None, batch_size=None):
    """
    Returns a generator which returns corresponding chunks of size batch_size
    for both inputs and additional_forward_args. If batch size is None,
    generator only includes original inputs and additional args.
    """
    assert batch_size is None or (
        isinstance(batch_size, int) and batch_size > 0
    ), "Batch size must be greater than 0."
    inputs = format_input(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    num_examples = inputs[0].shape[0]
    if batch_size is None:
        yield inputs, additional_forward_args
    else:
        current_total = 0
        while current_total < num_examples:
            yield _tuple_splice_range(
                inputs, current_total, current_total + batch_size
            ), _tuple_splice_range(
                additional_forward_args, current_total, current_total + batch_size
            )
            current_total += batch_size


def _batched_operator(
    operator, inputs, additional_forward_args=None, batch_size=None, **kwargs
):
    """
    Batches the operation of the given operator, applying the given batch size
    to inputs and additional forward arguments, and returning the concatenation
    of the results of each batch.
    """
    all_outputs = []
    for input, additional in _batched_generator(
        inputs, additional_forward_args, batch_size
    ):
        all_outputs.append(
            operator(inputs=input, additional_forward_args=additional, **kwargs)
        )
    return _reduce_list(all_outputs)
