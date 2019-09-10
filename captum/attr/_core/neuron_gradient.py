#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.common import (
    format_input,
    _format_additional_forward_args,
    _format_attributions,
)
from .._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    _forward_layer_eval,
)


class NeuronGradient(NeuronAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(
        self, inputs, neuron_index, additional_forward_args=None, device_ids=None
    ):
        r"""
            Computes gradient with respect to input of particular neuron in
            target hidden layer.

            Args

                inputs:     A single high dimensional input tensor, in which
                            dimension 0 corresponds to number of examples.
                neuron_index: Tuple providing index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).

            Return

                attributions: Activation of each neuron in output of given layer
        """
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)

        layer_out, input_grads = _forward_layer_eval(
            self.forward_func,
            inputs,
            self.layer,
            additional_forward_args,
            neuron_index,
            device_ids=self.device_ids,
        )

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, input_grads)
