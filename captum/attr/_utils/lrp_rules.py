#!/usr/bin/env python3

import copy

import torch
import torch.nn as nn


class PropagationRule:
    """
    Abstract class as basis for all propagation rule classes.
    STABILITY_FACTOR is used for assuring that no zero divison occurs.
    """

    STABILITY_FACTOR = 1e-9

    def forward_hook(self, module, inputs, outputs):
        """Function that registers backward hooks on input and output
        tensors of linear layers in the model."""
        module.outputs = outputs.data
        # print(f"Initial output: {torch.sum(outputs.data)}")
        input_hook = self._create_backward_hook_input(inputs[0].data)
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_input_hook = inputs[0].register_hook(input_hook)
        self._handle_output_hook = outputs.register_hook(output_hook)

    def backward_hook_activation(self, module, grad_input, grad_output):
        """Function that serves as backward hook to propagate relevance
        over non-linear activations without manipulation."""
        return grad_output

    def _backward_hook_relevance(self, module, grad_input, grad_output):
        module.relevance = grad_output[0] * module.outputs
        # print(f"Intermediate relevance: {torch.sum(module.relevance)}")

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            # print("run input hook")
            return grad * inputs

        return _backward_hook_input

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            # print("run output hook")
            return grad / (outputs + self.STABILITY_FACTOR)

        return _backward_hook_output


class PropagationRule_ManipulateModules(PropagationRule):
    def __init__(self, set_bias_to_zero=False):
        self.set_bias_to_zero = set_bias_to_zero

    def forward_hook_weights(self, module, inputs, outputs):
        """Save initial activations a_j before modules are changed"""
        module.activation = inputs[0].data
        self._manipulate_weights(module, inputs, outputs)

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


class Z_Rule(EpsilonRule):
    """
    Basic rule for relevance propagation, also called LRP-0,
    see Montavon et al. Explainable AI book.

    Use for upper layers.
    """


class GammaRule(PropagationRule_ManipulateModules):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (int): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma=0.25):
        super(GammaRule, self).__init__()
        self.gamma = gamma

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = (
                module.weight.data + self.gamma * module.weight.data.clamp(min=0)
            )
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class Alpha1_Beta0_Rule(PropagationRule_ManipulateModules):
    """
    Alpha1_Beta0 rule for relevance backpropagation, also known
    as Deep-Taylor. Only positive relevance is propagated, resulting
    in stable results, therefore recommended as the initial choice.

    Use for lower layers.
    """

    def __init__(self):
        super(Alpha1_Beta0_Rule, self).__init__()

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(min=0)
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class zB_Rule(PropagationRule):
    """
    Attention: Implementation is not using hooks!

    Propagation rule for pixel layers (first layer).
    If image is normalized, the mean and std need to be given.

    Args:
        minimum_value (int): Minimum value of feature space (typically 0 for pixels)
        maximum_value (int): Maximum value of feature space (typically 1 or 255 for pixels)
        mean (tensor): Tensor containing the mean values for all channels if images were normalized.
        std (tensor): Tensor containing the standard deviation values for all channels if images were normalized.
    """

    def __init__(self, minimum_value=0, maximum_value=1, mean=None, std=None):
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.mean = mean
        self.std = std

    def propagate(self, relevance, layer, activations):

        activations = (activations.data).requires_grad_(True)

        if self.mean is not None and self.std is not None:
            mean = torch.Tensor(self.mean).reshape(1, -1, 1, 1)
            std = torch.Tensor(self.std).reshape(1, -1, 1, 1)
            lb = (activations.data * 0 + (0 - mean) / std).requires_grad_(True)
            hb = (activations.data * 0 + (1 - mean) / std).requires_grad_(True)
        else:
            lb = (
                torch.Tensor(activations.size())
                .fill_(self.minimum_value)
                .requires_grad_(True)
            )
            hb = (
                torch.Tensor(activations.size())
                .fill_(self.maximum_value)
                .requires_grad_(True)
            )

        layer_copy = copy.deepcopy(layer)
        z = layer_copy.forward(activations) + self.STABILITY_FACTOR
        z = z - self._layer_copy_negative_weights(layer).forward(hb)
        z = z - self._layer_copy_positive_weights(layer).forward(lb)
        s = (relevance / z).data
        (z * s.data).sum().backward()
        c, cp, cm = activations.grad, lb.grad, hb.grad
        propagated_relevance = (activations * c + lb * cp + hb * cm).data

        return propagated_relevance

    def _layer_copy_negative_weights(self, layer):
        return self._layer_copy_filtered_weights(layer, lambda p: p.clamp(min=0))

    def _layer_copy_positive_weights(self, layer):
        return self._layer_copy_filtered_weights(layer, lambda p: p.clamp(max=0))

    def _layer_copy_filtered_weights(self, layer, filter_weights):
        layer = copy.deepcopy(layer)
        if hasattr(layer, "weight") and layer.weight is not None:
            new_weights = filter_weights(layer.weight.data)
            layer.weight.data = new_weights
        if hasattr(layer, "bias") and layer.bias is not None:
            new_bias = filter_weights(layer.bias.data)
            layer.bias.data = new_bias
        return layer


def suggested_rules(model):
    """
    Return a list of rules for a given model.

    Args:
        model (str): Name of the used model (vgg16)
    """
    if model == "vgg16":
        layer0 = [zB_Rule(0, 1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        layers1_16 = [GammaRule() for i in range(16)]
        layers17_30 = [EpsilonRule() for i in range(14)]
        layers31_38 = [EpsilonRule() for i in range(8)]
        rules = layer0 + layers1_16 + layers17_30 + layers31_38
    else:
        raise NotImplementedError("No suggested rules for given model")

    return rules
