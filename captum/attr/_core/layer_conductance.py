#!/usr/bin/env python3
import torch
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import LayerAttribution
from .._utils.batching import _batched_operator
from .._utils.common import (
    _reshape_and_sum,
    _format_input_baseline,
    _format_additional_forward_args,
    _expand_additional_forward_args,
    validate_input,
)
from .._utils.gradient import compute_layer_gradients_and_eval


class LayerConductance(LayerAttribution):
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
        method="riemann_trapezoid",
        batch_size=None,
    ):
        r"""
            Computes conductance using gradients along the path, applying
            riemann's method or gauss-legendre.
            The details of the approach can be found here:
            https://arxiv.org/abs/1805.12233
            https://arxiv.org/pdf/1807.09946.pdf

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

                attributions: Total conductance with respect to each neuron in
                              output of given layer
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        validate_input(inputs, baselines, n_steps, method)

        num_examples = inputs[0].shape[0]

        # Retrieve scaling factors for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        alphas = alphas_func(n_steps + 1)

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
        # dim -> (#examples * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps + 1)
            if additional_forward_args is not None
            else None
        )

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = _batched_operator(
            compute_layer_gradients_and_eval,
            scaled_features_tpl,
            input_additional_args,
            batch_size=batch_size,
            forward_fn=self.forward_func,
            layer=self.layer,
            target_ind=target,
            device_ids=self.device_ids,
        )

        # Compute differences between consecutive evaluations of layer_eval.
        # This approximates the total input gradient of each step multiplied
        # by the step size.
        grad_diffs = layer_eval[num_examples:] - layer_eval[:-num_examples]

        # Element-wise mutliply gradient of output with respect to hidden layer
        # and summed gradients with respect to input (chain rule) and sum
        # across stepped inputs.
        return _reshape_and_sum(
            grad_diffs * layer_gradients[:-num_examples],
            n_steps,
            num_examples,
            layer_eval.shape[1:],
        )
