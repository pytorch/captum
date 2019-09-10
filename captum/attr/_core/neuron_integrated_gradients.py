#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.gradient import _forward_layer_eval

from .integrated_gradients import IntegratedGradients


class NeuronIntegratedGradients(NeuronAttribution):
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
        neuron_index,
        baselines=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        batch_size=None,
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

        def grad_fn(forward_fn, inputs, target_ind=None, additional_forward_args=None):
            _, grads = _forward_layer_eval(
                forward_fn,
                inputs,
                self.layer,
                additional_forward_args,
                neuron_index,
                device_ids=self.device_ids,
            )
            return grads

        ig = IntegratedGradients(self.forward_func)
        ig.gradient_func = grad_fn
        # Return only attributions and not delta
        return ig.attribute(
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
            batch_size=batch_size,
        )[0]
