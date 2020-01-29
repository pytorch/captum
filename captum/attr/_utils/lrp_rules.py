#!/usr/bin/env python3

import copy

import torch
import torch.nn as nn

# TODO: Implement w^2 rule


class PropagationRule(object):
    """
    Abstract class as basis for all propagation rule classes.
    STABILITY_FACTOR is used for assuring that no zero divison occurs.
    """

    STABILITY_FACTOR = 1e-9

    def propagate(self, relevance, layer, activations):
        raise NotImplementedError


class EpsilonRhoRule(PropagationRule):
    """
    General propagation rule which is basis for many special cases.
    Implementation of epsilon-rho-rule according to Montavon et al.
    in Explainable AI book

    Args:
        rho (callable): Function by which the layer weights and bias is manipulated
            during propagation.
            Example:
                def _rho_function(tensor_input):
                    tensor_positive = torch.clamp(tensor_input, min=0)
                    tensor_output = tensor_input + (0.25 * tensor_positive)
                    return tensor_output
        epsilon (callable): Function to manipulate z-values during propagation
            example: lambda z: z + 1e-9
    """

    def __init__(
        self, rho=lambda p: p, epsilon=lambda z: z + PropagationRule.STABILITY_FACTOR
    ):
        self.epsilon = epsilon
        self.rho = rho

    def propagate(self, relevance, layer, activations):

        activations = (activations.data).requires_grad_(True)
        layer_copy = copy.deepcopy(layer)
        layer_rho = self._apply_rho(layer_copy)
        z = self.epsilon(layer_rho.forward(activations))
        # Correct shape when tensor is flattened
        if relevance.shape != z.shape:
            relevance = relevance.view(z.shape)
        s = (relevance / z).data
        (z * s.data).sum().backward()
        c = activations.grad
        propagated_relevance = (activations * c).da
        ta

        return propagated_relevance

    def _apply_rho(self, layer):
        if hasattr(layer, "weight") and layer.weight is not None:
            new_weights = self.rho(layer.weight.data)
            layer.weight.data = new_weights
        if hasattr(layer, "bias") and layer.bias is not None:
            new_bias = self.rho(layer.bias.data)
            layer.bias.data = new_bias
        return layer


class BasicRule(EpsilonRhoRule):
    """
    Basic rule for relevance propagation, also called LRP-0,
    see Montavon et al. Explainable AI book.
    Implemented using the EpsilonRhoRule.

    Use for upper layers.
    """

    def __init__(self):
        super(BasicRule, self).__init__()


class EpsilonRule(EpsilonRhoRule):
    """
    Rule for relevance propagation using a small value of epsilon
    to avoid numerical instabilities and remove noise.

    Use for middle layers.

    Args:
        epsilon (callable): Function to manipulate z-values during propagation
            example: lambda z: z + 1e-9
    """

    def __init__(self, epsilon):
        super(EpsilonRule, self).__init__(epsilon=epsilon)


class GammaRule(EpsilonRhoRule):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (int): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma=0.25):
        def _gamma_function(tensor_input):
            tensor_positive = torch.clamp(tensor_input, min=0)
            tensor_output = tensor_input + (gamma * tensor_positive)
            return tensor_output

        super(GammaRule, self).__init__(rho=_gamma_function)


class Alpha1_Beta0_Rule(EpsilonRhoRule):
    """
    Alpha1_Beta0 rule for relevance backpropagation, also known
    as Deep-Taylor. Only positive relevance is propagated, resulting
    in stable results, therefore recommended as the initial choice.

    Use for lower layers.
    """

    def __init__(self):
        def _rho_function(tensor_input):
            tensor_output = torch.clamp(tensor_input, min=0)
            return tensor_output

        super(Alpha1_Beta0_Rule, self).__init__(rho=_rho_function)


class zB_Rule(PropagationRule):
    """
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
        layers1_16 = [GammaRule(gamma=0.25) for i in range(16)]
        layers17_30 = [
            EpsilonRule(
                epsilon=lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** 0.5).data
            )
            for i in range(14)
        ]
        layers31_38 = [EpsilonRule(epsilon=lambda z: z + 1e-9) for i in range(8)]
        rules = layer0 + layers1_16 + layers17_30 + layers31_38
    else:
        raise NotImplementedError("No suggested rules for given model")

    return rules
