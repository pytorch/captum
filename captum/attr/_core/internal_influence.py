#!/usr/bin/env python3
import torch
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import LayerAttribution
from .._utils.batching import _batched_operator
from .._utils.common import (
    _reshape_and_sum,
    _format_input_baseline,
    validate_input,
    _format_additional_forward_args,
    _expand_additional_forward_args,
)
from .._utils.gradient import compute_layer_gradients_and_eval


class InternalInfluence(LayerAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        batch_size=None,
    ):
        r"""
            Computes internal influence using gradients along the path, applying
            Riemann's method or Gauss-Legendre. This is effectively similar to
            applying integrated gradients on the given layer.

            More details on this approach can be found here:
            https://arxiv.org/pdf/1802.03788.pdf

            Args

                inputs:     A single high dimensional input tensor, in which
                            dimension 0 corresponds to number of examples.
                baselines:   A single high dimensional baseline tensor,
                            which has the same shape as the input
                target:     Predicted class index. This is necessary only for
                            classification use cases
                n_steps:    The number of steps used by the approximation method
                method:     Method for integral approximation, one of `riemann_right`,
                            `riemann_left`, `riemann_middle`, `riemann_trapezoid`
                            or `gausslegendre`

            Return

                attributions: Total internal influence with respect to each neuron in
                              output of given layer
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        validate_input(inputs, baselines, n_steps, method)

        # Retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # Compute scaled inputs from baseline to final input.
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguemnts
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        # Returns gradient of output with respect to hidden layer.
        layer_gradients, _ = _batched_operator(
            compute_layer_gradients_and_eval,
            scaled_features_tpl,
            input_additional_args,
            batch_size=batch_size,
            forward_fn=self.forward_func,
            layer=self.layer,
            target_ind=target,
            device_ids=self.device_ids,
        )
        # flattening grads so that we can multipy it with step-size
        # calling contigous to avoid `memory whole` problems
        scaled_grads = layer_gradients.contiguous().view(n_steps, -1) * torch.tensor(
            step_sizes
        ).view(n_steps, 1).to(layer_gradients.device)

        # aggregates across all steps for each tensor in the input tuple
        return _reshape_and_sum(
            scaled_grads, n_steps, inputs[0].shape[0], layer_gradients.shape[1:]
        )
