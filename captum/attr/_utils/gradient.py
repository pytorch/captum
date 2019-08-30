#!/usr/bin/env python3
import torch

from .common import _run_forward


def prepare_gradient_inputs(input_tensors):
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.
    """
    assert isinstance(
        input_tensors, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients"
    grad_required = []
    for tensor in input_tensors:
        assert isinstance(tensor, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(tensor.requires_grad)
        tensor.requires_grad_()
        if tensor.grad is not None:
            tensor.grad.zero_()
    return grad_required


def undo_gradient_requirements(input_tensors, grad_required):
    """
    Iterates through list of tensors, zeros each gradient, and sets required
    grad to false if the corresponding index in grad_required is False.
    This method is used to undo the effects of prepare_gradient_inputs, making
    grads not required for any input tensor that did not initially require
    gradients.
    """

    assert isinstance(
        input_tensors, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients."
    assert len(input_tensors) == len(
        grad_required
    ), "Input tuple length should match gradient mask."
    for index, tensor in enumerate(input_tensors):
        assert isinstance(tensor, torch.Tensor), "Given input is not a torch.Tensor"
        if tensor.grad is not None:
            tensor.grad.zero_()
        if not grad_required[index]:
            tensor.requires_grad_(False)


def compute_gradients(forward_fn, input, target_ind=None, additional_forward_args=None):
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
        output = _run_forward(forward_fn, input, target_ind, additional_forward_args)

        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(output), input)
    return grads


def compute_layer_gradients_and_eval(
    forward_fn, layer, inputs, target_ind=None, additional_forward_args=None
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
        saved_layer_output = None

        # Set a forward hook on specified module and run forward pass to
        # get layer output tensor.
        def forward_hook(module, inp, out):
            nonlocal saved_layer_output
            saved_layer_output = out

        hook = layer.register_forward_hook(forward_hook)
        output = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        # Remove unnecessary forward hook.
        hook.remove()
        saved_grads = torch.autograd.grad(torch.unbind(output), saved_layer_output)[0]

    return saved_grads, saved_layer_output
