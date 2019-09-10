#!/usr/bin/env python3
import threading
import torch
import warnings

from .common import _run_forward, _extend_index_list
from .batching import _reduce_list, _sort_key_list


def apply_gradient_requirements(inputs):
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.
    """
    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients"
    grad_required = []
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(input.requires_grad)
        if not input.requires_grad:
            warnings.warn(
                "Input Tensor %d did not already require gradients, "
                "required_grads has been set automatically." % index
            )
            input.requires_grad_()
        if input.grad is not None:
            if torch.sum(torch.abs(input.grad)).item() > 1e-7:
                warnings.warn(
                    "Input Tensor %d had a non-zero gradient tensor, "
                    "which is being reset to 0." % index
                )
            input.grad.zero_()
    return grad_required


def undo_gradient_requirements(inputs, grad_required):
    """
    Iterates through list of tensors, zeros each gradient, and sets required
    grad to false if the corresponding index in grad_required is False.
    This method is used to undo the effects of prepare_gradient_inputs, making
    grads not required for any input tensor that did not initially require
    gradients.
    """

    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients."
    assert len(inputs) == len(
        grad_required
    ), "Input tuple length should match gradient mask."
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        if input.grad is not None:
            input.grad.detach_()
            input.grad.zero_()
        if not grad_required[index]:
            input.requires_grad_(False)


def compute_gradients(
    forward_fn, inputs, target_ind=None, additional_forward_args=None, batch_size=None
):
    r"""
        Computes gradients of the output with respect to inputs for an
        arbitrary forward function.

        Args

            forward_fn: forward function. This can be for example model's
                        forward function.
            input:      Input at which gradients are evaluated,
                        will be passed to forward_fn.
            target_ind: Index of the target class for which gradients
                        must be computed (classification only).
            args:       Additional input arguments that forward function requires.
                        It takes an empty tuple (no additional arguments) if no
                        additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        output = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)

        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(output), inputs)
    return grads


def _neuron_gradients(inputs, saved_layer_outputs, key_list, gradient_neuron_index):
    with torch.autograd.set_grad_enabled(True):
        gradient_tensors = []
        for key in key_list:
            current_out_tensor = saved_layer_outputs[key]
            gradient_tensors.append(
                torch.autograd.grad(
                    [
                        current_out_tensor[index]
                        for index in _extend_index_list(
                            current_out_tensor.shape[0], gradient_neuron_index
                        )
                    ],
                    inputs,
                )
            )
        return _reduce_list(gradient_tensors, sum)


def _forward_layer_eval(
    forward_fn,
    inputs,
    layer,
    additional_forward_args=None,
    gradient_neuron_index=None,
    device_ids=None,
):
    saved_layer_outputs = {}
    lock = threading.Lock()

    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor.
    def forward_hook(module, inp, out):
        assert isinstance(
            out, torch.Tensor
        ), "Layers with multiple output tensors are not yet supported."
        with lock:
            nonlocal saved_layer_outputs
            saved_layer_outputs[out.device] = out

    hook = layer.register_forward_hook(forward_hook)
    _run_forward(forward_fn, inputs, additional_forward_args=additional_forward_args)
    hook.remove()

    if len(saved_layer_outputs) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    if len(saved_layer_outputs) > 1 and device_ids is None:
        if (
            isinstance(forward_fn, torch.nn.DataParallel)
            and forward_fn.device_ids is not None
        ):
            device_ids = forward_fn.device_ids
        else:
            raise AssertionError(
                "DataParallel Model Detected, device ID list or DataParallel model"
                " must be provided for identifying device batch ordering."
            )

    key_list = _sort_key_list(list(saved_layer_outputs.keys()), device_ids)
    if gradient_neuron_index is not None:
        inp_grads = _neuron_gradients(
            inputs, saved_layer_outputs, key_list, gradient_neuron_index
        )
        return torch.cat([saved_layer_outputs[dev] for dev in key_list]), inp_grads
    else:
        return torch.cat([saved_layer_outputs[dev] for dev in key_list])


def compute_layer_gradients_and_eval(
    forward_fn,
    layer,
    inputs,
    target_ind=None,
    additional_forward_args=None,
    gradient_neuron_index=None,
    device_ids=None,
):
    r"""
        Computes gradients of the output with respect to a given layer as well
        as the output evaluation of the layer for an arbitrary forward function
        and given input.

        Args

            forward_fn: forward function. This can be for example model's
                        forward function.
            layer:      Layer for which gradients / output will be evaluated.
            inputs:     Input at which gradients are evaluated,
                        will be passed to forward_fn.
            target_ind: Index of the target class for which gradients
                        must be computed (classification only).
            args:       Additional input arguments that forward function requires.
                        It takes an empty tuple (no additional arguments) if no
                        additional arguments are required


        Return

            gradients:  Gradients of output with respect to target layer output.
            evals:      Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        saved_layer_outputs = {}
        lock = threading.Lock()

        # Set a forward hook on specified module and run forward pass to
        # get layer output tensor.
        def forward_hook(module, inp, out):
            with lock:
                assert isinstance(
                    out, torch.Tensor
                ), "Layers with multiple output tensors are not yet supported."
                nonlocal saved_layer_outputs
                saved_layer_outputs[out.device] = out

        hook = layer.register_forward_hook(forward_hook)
        output = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        # Remove unnecessary forward hook.
        hook.remove()

        if len(saved_layer_outputs) == 0:
            raise AssertionError(
                "Forward hook did not obtain any outputs for given layer"
            )

        if len(saved_layer_outputs) > 1 and device_ids is None:
            if (
                isinstance(forward_fn, torch.nn.DataParallel)
                and forward_fn.device_ids is not None
            ):
                device_ids = forward_fn.device_ids
            else:
                raise AssertionError(
                    "DataParallel Model Detected, device ID list or DataParallel model"
                    " must be provided for identifying device batch ordering."
                )

        key_list = _sort_key_list(list(saved_layer_outputs.keys()), device_ids)
        all_outputs = torch.cat([saved_layer_outputs[dev] for dev in key_list])
        grad_inputs = tuple(saved_layer_outputs[dev] for dev in key_list)
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs)
        all_grads = torch.cat(saved_grads)
        if gradient_neuron_index is not None:
            inp_grads = _neuron_gradients(
                inputs, saved_layer_outputs, key_list, gradient_neuron_index
            )
            return all_grads, all_outputs, inp_grads
        else:
            return all_grads, all_outputs
