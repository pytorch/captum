#!/usr/bin/env python3
import threading
import torch
import warnings

from .common import _run_forward, _verify_select_column
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
    forward_fn, inputs, target_ind=None, additional_forward_args=None
):
    r"""
        Computes gradients of the output with respect to inputs for an
        arbitrary forward function.

        Args:

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
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(output), inputs)
    return grads


def _neuron_gradients(inputs, saved_layer, key_list, gradient_neuron_index):
    with torch.autograd.set_grad_enabled(True):
        gradient_tensors = []
        for key in key_list:
            current_out_tensor = saved_layer[key]
            gradient_tensors.append(
                torch.autograd.grad(
                    torch.unbind(
                        _verify_select_column(current_out_tensor, gradient_neuron_index)
                    ),
                    inputs,
                )
            )
        return _reduce_list(gradient_tensors, sum)


def _forward_layer_eval(
    forward_fn,
    inputs,
    layer,
    additional_forward_args=None,
    device_ids=None,
    attribute_to_layer_input=False,
):
    return _forward_layer_eval_with_neuron_grads(
        forward_fn,
        inputs,
        layer,
        additional_forward_args=additional_forward_args,
        gradient_neuron_index=None,
        device_ids=device_ids,
        attribute_to_layer_input=attribute_to_layer_input,
    )


def _forward_layer_eval_with_neuron_grads(
    forward_fn,
    inputs,
    layer,
    additional_forward_args=None,
    gradient_neuron_index=None,
    device_ids=None,
    attribute_to_layer_input=False,
):
    """
    This method computes forward evaluation for a particular layer using a
    forward hook. If a gradient_neuron_index is provided, then gradients with
    respect to that neuron in the layer output are also returned.

    These functionalities are combined due to the behavior of DataParallel models
    with hooks, in which hooks are executed once per device. We need to internally
    combine the separated tensors from devices by concatenating based on device_ids.
    Any necessary gradients must be taken with respect to each independent batched
    tensor, so the gradients are computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel models
    can be found in the PyTorch data parallel documentation. We maintain the separate
    evals in a dictionary protected by a lock, analogous to the gather implementation
    for the core PyTorch DataParallel implementation.
    """
    saved_layer = {}
    lock = threading.Lock()

    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor(s).
    # For DataParallel models, each partition adds entry to dictionary
    # with key as device and value as corresponding Tensor.
    def forward_hook(module, inp, out):
        eval_tsr = inp if attribute_to_layer_input else out

        # if `inp` or `out` is a tuple of one tensor, assign that tensor to `eval_tsr`
        if isinstance(eval_tsr, tuple) and len(eval_tsr) == 1:
            eval_tsr = eval_tsr[0]

        assert isinstance(
            eval_tsr, torch.Tensor
        ), "Layers with multiple inputs or output tensors are not supported yet."
        with lock:
            nonlocal saved_layer
            # TODO we need to think what will be the best way of storing eval
            # tensors per device for each input per example. This implementation
            # doesn't support a tuple of inputs
            saved_layer[eval_tsr.device] = eval_tsr

    hook = layer.register_forward_hook(forward_hook)
    _run_forward(forward_fn, inputs, additional_forward_args=additional_forward_args)
    hook.remove()
    if len(saved_layer) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    # Multiple devices / keys implies a DataParallel model, so we look for
    # device IDs if given or available from forward function
    # (DataParallel model object).
    if len(saved_layer) > 1 and device_ids is None:
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

    # Identifies correct device ordering based on device ids.
    # key_list is a list of devices in appropriate ordering for concatenation.
    # If only one key exists (standard model), key list simply has one element.
    key_list = _sort_key_list(list(saved_layer.keys()), device_ids)
    if gradient_neuron_index is not None:
        inp_grads = _neuron_gradients(
            inputs, saved_layer, key_list, gradient_neuron_index
        )
        return (
            torch.cat([saved_layer[device_id] for device_id in key_list]),
            inp_grads,
        )
    else:
        return torch.cat([saved_layer[device_id] for device_id in key_list])


def compute_layer_gradients_and_eval(
    forward_fn,
    layer,
    inputs,
    target_ind=None,
    additional_forward_args=None,
    gradient_neuron_index=None,
    device_ids=None,
    attribute_to_layer_input=False,
):
    r"""
        Computes gradients of the output with respect to a given layer as well
        as the output evaluation of the layer for an arbitrary forward function
        and given input.

        For data parallel models, hooks are executed once per device ,so we
        need to internally combine the separated tensors from devices by
        concatenating based on device_ids. Any necessary gradients must be taken
        with respect to each independent batched tensor, so the gradients are
        computed and combined appropriately.

        More information regarding the behavior of forward hooks with DataParallel
        models can be found in the PyTorch data parallel documentation. We maintain
        the separate inputs in a dictionary protected by a lock, analogous to the
        gather implementation for the core PyTorch DataParallel implementation.

        Args:

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


        Returns:
            2-element tuple of **gradients**, **evals**:
            - **gradients**:
                Gradients of output with respect to target layer output.
            - **evals**:
                Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        saved_layer = {}
        lock = threading.Lock()

        # Set a forward hook on specified module and run forward pass to
        # get layer output tensor(s).
        # For DataParallel models, each partition adds entry to dictionary
        # with key as device and value as corresponding Tensor.
        def forward_hook(module, inp, out):
            eval_tsr = inp if attribute_to_layer_input else out

            # if `inp` or `out` is a tuple of one tensor, assign that
            # tensor to `eval_tsr`
            if isinstance(eval_tsr, tuple) and len(eval_tsr) == 1:
                eval_tsr = eval_tsr[0]

            with lock:
                assert isinstance(eval_tsr, torch.Tensor), (
                    "Layers with multiple input or output tensors are not"
                    "yet supported."
                )
                nonlocal saved_layer
                saved_layer[eval_tsr.device] = eval_tsr

        hook = layer.register_forward_hook(forward_hook)
        output = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # Remove unnecessary forward hook.
        hook.remove()

        if len(saved_layer) == 0:
            raise AssertionError(
                "Forward hook did not obtain any outputs for given layer"
            )

        # Multiple devices / keys implies a DataParallel model, so we look for
        # device IDs if given or available from forward function
        # (DataParallel model object).
        if len(saved_layer) > 1 and device_ids is None:
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

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(list(saved_layer.keys()), device_ids)
        all_outputs = _reduce_list([saved_layer[device_id] for device_id in key_list])
        grad_inputs = tuple(saved_layer[device_id] for device_id in key_list)
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs)
        all_grads = torch.cat(saved_grads)
        if gradient_neuron_index is not None:
            inp_grads = _neuron_gradients(
                inputs, saved_layer, key_list, gradient_neuron_index
            )
            return all_grads, all_outputs, inp_grads
        else:
            return all_grads, all_outputs


def construct_neuron_grad_fn(layer, neuron_index, device_ids):
    def grad_fn(forward_fn, inputs, target_ind=None, additional_forward_args=None):
        _, grads = _forward_layer_eval_with_neuron_grads(
            forward_fn,
            inputs,
            layer,
            additional_forward_args,
            neuron_index,
            device_ids=device_ids,
        )
        return grads

    return grad_fn
