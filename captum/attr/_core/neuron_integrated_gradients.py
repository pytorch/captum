#!/usr/bin/env python3
import torch
from .._utils.attribution import NeuronAttribution
from .._utils.common import _forward_layer_eval, _extend_index_list

from .integrated_gradients import IntegratedGradients


class NeuronIntegratedGradients(NeuronAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def attribute(
        self, inputs, neuron_index, baselines=None, n_steps=50, additional_forward_args=None, method="gausslegendre"
    ):
        r"""
            Computes integrated gradients for a particular neuron in the given
            target layer, applying Riemann's method or Gauss-Legendre.

            More details on this approach can be found here:
            https://arxiv.org/pdf/1802.03788.pdf

            Args

                inputs:     A single high dimensional input tensor, in which
                            dimension 0 corresponds to number of examples.
                neuron_index: Tuple providing index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).
                baselines:   A single high dimensional baseline tensor,
                            which has the same shape as the input
                n_steps:    The number of steps used by the approximation method
                method:     Method for integral approximation, one of `riemann_right`,
                            `riemann_left`, `riemann_middle`, `riemann_trapezoid`
                            or `gausslegendre`

            Return

                attributions: Total integrated gradients with respect to particular
                              neuron in output of given layer. Output size matches
                              that of the input.
        """

        def forward_fn(*args):
            layer_output = _forward_layer_eval(
                self.forward_func, args, self.layer
            )
            indices = _extend_index_list(args[0].shape[0], neuron_index)
            return torch.stack(tuple(layer_output[i] for i in indices))

        ig = IntegratedGradients(forward_fn)
        # Return only attributions and not delta
        return ig.attribute(inputs, baselines, additional_forward_args=additional_forward_args, n_steps=n_steps, method=method)[0]
