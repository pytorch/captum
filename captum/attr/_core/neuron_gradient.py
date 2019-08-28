#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.common import _forward_layer_eval, _extend_index_list

import torch


class NeuronGradient(NeuronAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def attribute(self, inputs, neuron_index):
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
        layer_out = _forward_layer_eval(self.forward_func, inputs, self.layer)
        indices = _extend_index_list(inputs.shape[0], neuron_index)
        with torch.autograd.set_grad_enabled(True):
            input_grads = torch.autograd.grad(
                [layer_out[index] for index in indices], inputs
            )[0]
        return input_grads
