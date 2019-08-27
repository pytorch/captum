#!/usr/bin/env python3
import torch
from .utils.approximation_methods import approximation_parameters
from .utils.attribution import LayerAttribution
from .utils.common import _reshape_and_sum
from .utils.gradient import compute_layer_gradients_and_eval


class Conductance(LayerAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        n_steps=500,
        method="riemann_trapezoid",
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
        if baselines is None:
            baselines = 0

        num_examples = inputs.shape[0]

        # Retrieve scaling factors for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        alphas = alphas_func(n_steps + 1)

        # Compute scaled inputs from baseline to final input.
        scaled_features = torch.cat(
            [baselines + alpha * (inputs - baselines) for alpha in alphas], dim=0
        )

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = compute_layer_gradients_and_eval(
            self.forward_func, self.layer, scaled_features, target
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
