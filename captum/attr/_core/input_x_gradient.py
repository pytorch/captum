#!/usr/bin/env python3
from .._utils.common import format_input, _format_attributions
from .._utils.attribution import GradientBasedAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


class InputXGradient(GradientBasedAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func:  The forward function of the model or any modification of it
        """
        super().__init__(forward_func)

    def attribute(self, inputs, target=None, additional_forward_args=None):
        r""""
        A baseline approach for computing the attribution. It multiplies input with
        its gradient.
        https://arxiv.org/abs/1611.07270

        Args:

                inputs:     A single high dimensional input tensor or a tuple of them.
                target:     Predicted class index. This is necessary only for
                            classification use cases
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
        gradient_mask = apply_gradient_requirements(inputs)
        additional_forward_args = (
            additional_forward_args if additional_forward_args else []
        )
        gradients = self.gradient_func(
            self.forward_func, inputs, target, *additional_forward_args
        )

        attributions = tuple(
            input * gradient for input, gradient in zip(inputs, gradients)
        )
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, attributions)
