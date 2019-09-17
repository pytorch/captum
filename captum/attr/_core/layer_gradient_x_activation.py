#!/usr/bin/env python3
from .._utils.attribution import LayerAttribution
from .._utils.common import format_input, _format_additional_forward_args
from .._utils.gradient import compute_layer_gradients_and_eval


class LayerGradientXActivation(LayerAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(self, inputs, target=None, additional_forward_args=None):
        r"""
            Computes activation of selected layer for given input.

            Args

                inputs:     A single high dimensional input tensor, in which
                            dimension 0 corresponds to number of examples.
                target:     Predicted class index. This is necessary only for
                            classification use cases

            Return

                attributions: Activation of each neuron in output of given layer
        """
        inputs = format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
        )
        return layer_gradients * layer_eval
