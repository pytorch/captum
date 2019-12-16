#!/usr/bin/env python3

from itertools import repeat

import torch
import torch.nn as nn

from .._utils.common import _format_attributions, _format_input
from .._utils.attribution import Attribution, GradientAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements

from .lrp_rules import PropagationRule, BasicRule

class LRP(GradientAttribution):
    def __init__(self, model, rules):
        """
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
            rules (iterator of PropagationRules): List of Rules for each layer
                        of forward_func
        """
        self.layers = self._get_layers(model)
        self.layer_count = len(self.layers)
        for rule in rules:
            if not isinstance(rule, PropagationRule):
                raise TypeError("Please select propagation rules inherited from class PropagationRule")
        self.rules = rules
        super(LRP, self).__init__(model)


    def attribute(self, inputs, relevance, **kwargs):
        activations = self._get_activations(inputs)
        for layer, rule, activation in zip(self.layers, self.rules, activations[1:]):
            # Convert Max-Pooling to Average Pooling layer
            if isinstance(layer, torch.nn.MaxPool2d):
                layer = torch.nn.AvgPool2d(2)
            # Propagate relevance for Conv2D and Pooling
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.AvgPool2d):
                relevance = rule.propagate(relevance, layer, activation)
            else:
                pass
        return relevance


    def _get_layers(self, forward_func):
        '''
        Get list of children modules of the forward function or model.
        '''
        layers = list(forward_func.children())
        return layers


    def _get_activations(self, inputs):
        '''
        Calculate activations for inputs for every layer of forward_func.
        '''
        activations = [inputs] + [None] * (self.layer_count)
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):# and isinstance(layers[index-1], nn.poo
                activations[index + 1] = layer.forward(activations[index].view(-1, layer.in_features))
            else:
                activations[index + 1] = layer.forward(activations[index])
        return activations


class LRP_0(LRP):
    """LRP class, which uses the base rule for every layer.

    Arguments:
        LRP {[type]} -- [description]
    """
    def __init__(self, forward_func):
        rules = repeat(BasicRule(), 1000)
        super(LRP_0, self).__init__(forward_func, rules)

