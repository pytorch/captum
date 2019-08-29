#!/usr/bin/env python3
import torch
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import LayerAttribution
from .._utils.common import _reshape_and_sum
from .._utils.gradient import compute_layer_gradients_and_eval


class InternalInfluence(LayerAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def attribute(
        self, inputs, baselines=None, target=None, n_steps=50, method="gausslegendre"
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
        if baselines is None:
            baselines = 0

        # Retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # Compute scaled inputs from baseline to final input.
        scaled_features = torch.cat(
            [baselines + alpha * (inputs - baselines) for alpha in alphas], dim=0
        )

        # Returns gradient of output with respect to hidden layer.
        layer_gradients, _ = compute_layer_gradients_and_eval(
            self.forward_func, self.layer, scaled_features, target
        )
        # flattening grads so that we can multipy it with step-size
        # calling contigous to avoid `memory whole` problems
        scaled_grads = layer_gradients.contiguous().view(n_steps, -1) * torch.tensor(
            step_sizes
        ).view(n_steps, 1).to(layer_gradients.device)

        # aggregates across all steps for each tensor in the input tuple
        return _reshape_and_sum(
            scaled_grads, n_steps, inputs.shape[0], layer_gradients.shape[1:]
        )
