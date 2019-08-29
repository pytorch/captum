#!/usr/bin/env python3

import torch

from .._utils.common import format_input, _format_attributions
from .._utils.attribution import GradientBasedAttribution


class Saliency(GradientBasedAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func:  The forward function of the model or any modification of it
        """
        super().__init__(forward_func)

    def attribute(self, inputs, target=None, abs=True, additional_forward_args=None):
        r""""
        A baseline approach for computing the attribution. It returns
        values of the gradients with respect to input.
        https://arxiv.org/pdf/1312.6034.pdf

        Args:

                inputs:     A single high dimensional input tensor or a tuple of them.
                target:     Predicted class index. This is necessary only for
                            classification use cases
                abs:        Returns absolute value of gradients if set to True,
                            otherwise returns the (signed) gradients if False.
                            Defalut: True
                additional_forward_args: Apart input tensor an additional input can be
                                         passed to forward function. It can have
                                         arbitrary length.


        Returns:

                attributions: input * gradients(with respect to each input features)

        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)

        additional_forward_args = (
            additional_forward_args if additional_forward_args else []
        )
        gradients = self.gradient_func(
            self.forward_func, inputs, target, *additional_forward_args
        )
        if abs:
            attributions = tuple(torch.abs(gradient) for gradient in gradients)
        else:
            attributions = gradients

        return _format_attributions(is_inputs_tuple, attributions)
