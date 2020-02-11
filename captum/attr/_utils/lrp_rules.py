#!/usr/bin/env python3

import copy

import torch
import torch.nn as nn


class PropagationRule:
    """
    Base class for all propagation rule classes, also called Z-Rule.
    STABILITY_FACTOR is used to make sure that no zero divison occurs.
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

    @staticmethod
    def backward_hook_activation(module, grad_input, grad_output):
        """Function that serves as backward hook to propagate relevance
        over non-linear activations without manipulation."""
        if type(module) == nn.BatchNorm2d:
            #print(f"{module}: input shape {list(input.shape for input in grad_input)} output shape: {list(output.shape for output in grad_output)}")
            #print(f"len of output: {len((grad_output[0], *grad_input[1:]))}")
            return (grad_output[0], *grad_input[1:])
        else:
            return grad_output

    def _backward_hook_relevance(self, module, grad_input, grad_output):
        module.relevance = grad_output[0] * module.outputs
        print(f"Intermediate relevance: {torch.sum(module.relevance)},\n layer: {module}")

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


def suggested_rules(model):
    """
    Return a list of rules for a given model.

    Args:
        model (str): Name of the used model (vgg16)
    """
    if model == "vgg16":
        layer0 = [
            EpsilonRule()
        ]  # [zB_Rule(0, 1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        layers1_16 = [GammaRule() for i in range(16)]
        layers17_30 = [EpsilonRule() for i in range(14)]
        layers31_38 = [EpsilonRule() for i in range(8)]
        rules = layer0 + layers1_16 + layers17_30 + layers31_38
        rules = dict(zip(range(len(rules)), rules))
    else:
        raise NotImplementedError("No suggested rules for given model")

    return rules
