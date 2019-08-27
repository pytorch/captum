#!/usr/bin/env python3
import torch

from .common import _run_forward


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


def compute_layer_gradients_and_eval(forward_fn, layer, input, target_ind=None):
    r"""
        Computes gradients of the output with respect to a given layer as well
        as the output evaluation of the layer for an arbitrary forward function
        and given input.

        Args

            forward_fn: forward function. This can be for example model's
                        forward function.
            layer:      Layer for which gradients / output will be evaluated.
            input:      Input at which gradients are evaluated,
                        will be passed to forward_fn.
            target_ind: Index of the target class for which gradients
                        must be computed (classification only).


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
        output = forward_fn(input)
        output = output[:, target_ind] if target_ind is not None else output
        # Remove unnecessary forward hook.
        hook.remove()
        saved_grads = torch.autograd.grad(torch.unbind(output), saved_layer_output)[0]

    return saved_grads, saved_layer_output
