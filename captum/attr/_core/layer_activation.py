#!/usr/bin/env python3
from .._utils.attribution import LayerAttribution
from .._utils.gradient import _forward_layer_eval


class LayerActivation(LayerAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def attribute(self, inputs, additional_forward_args=None, device_ids=None):
        r"""
            Computes activation of selected layer for given input.

            Args

                inputs:     A single high dimensional input tensor, in which
                            dimension 0 corresponds to number of examples.

            Return

                attributions: Activation of each neuron in output of given layer
        """
        return _forward_layer_eval(
            self.forward_func,
            inputs,
            self.layer,
            additional_forward_args,
            device_ids=device_ids,
        )
