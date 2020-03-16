#!/usr/bin/env python3

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PropagationRule(ABC):
    """
    Base class for all propagation rule classes, also called Z-Rule.
    STABILITY_FACTOR is used to assure that no zero divison occurs.
    """

    STABILITY_FACTOR = 1e-9

    def forward_hook(self, module, inputs, outputs):
        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        module.outputs = outputs.data
        input_hook = self._create_backward_hook_input(inputs[0].data)
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_input_hook = inputs[0].register_hook(input_hook)
        self._handle_output_hook = outputs.register_hook(output_hook)

    @staticmethod
    def backward_hook_activation(module, grad_input, grad_output):
        """Backward hook to propagate relevance over non-linear activations."""
        return grad_output

    def _backward_hook_relevance(self, module, grad_input, grad_output):
        module.relevance = grad_output[0] * module.outputs
        print(f"Intermediate relevance: {torch.sum(module.relevance)}, layer: {module}")

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            return grad * inputs

        return _backward_hook_input

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            return grad / (outputs + torch.sign(outputs) * self.STABILITY_FACTOR)

        return _backward_hook_output

    def forward_hook_weights(self, module, inputs, outputs):
        """Save initial activations a_j before modules are changed"""
        module.activation = inputs[0].data
        self._manipulate_weights(module, inputs, outputs)

    @abstractmethod
    def _manipulate_weights(self, module, inputs, outputs):
        raise NotImplementedError

    def forward_pre_hook_activations(self, module, inputs):
        """Pass initial activations to graph generation pass"""
        inputs[0].data = module.activation.data
        return inputs


class EpsilonRule(PropagationRule):
    """
    Rule for relevance propagation using a small value of epsilon
    to avoid numerical instabilities and remove noise.

    Use for middle layers.

    Args:
        epsilon (integer, float): Value by which is added to the
        discriminator during propagation.
    """

    def __init__(self, epsilon=1e-9):
        self.STABILITY_FACTOR = epsilon

    def _manipulate_weights(self, module, inputs, outputs):
        pass


class GammaRule(PropagationRule):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (int): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma=0.25, set_bias_to_zero=False):
        self.gamma = gamma
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = (
                module.weight.data + self.gamma * module.weight.data.clamp(min=0)
            )
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class Alpha1_Beta0_Rule(PropagationRule):
    """
    Alpha1_Beta0 rule for relevance backpropagation, also known
    as Deep-Taylor. Only positive relevance is propagated, resulting
    in stable results, therefore recommended as the initial choice.

    Use for lower layers.
    """

    def __init__(self, set_bias_to_zero=False):
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(min=0)
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
