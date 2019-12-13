#!/usr/bin/env python3

from itertools import repeat

import torch
import torch.nn as nn

from .._utils.common import _format_attributions, _format_input
from .._utils.attribution import Attribution, GradientAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements

import math

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
        self.rules = rules
        for rule in rules:
            assert isinstance(rule, PropagationRule), "Please select propagation rules inherited from class PropagationRule"
        super(LRP, self).__init__(model)


    def attribute(self, inputs, relevance, **kwargs):
        activations = self._get_activations(inputs)
        for layer, rule in zip(self.layers, self.rules):
            relevance = rule.propagate(relevance, layer, activations)

        return relevance

    def _get_layers(self, forward_func):
        '''
        Get list of all feature and classifier layers of the forward function or model.
        '''
        children = list(forward_func.children())
        return children


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
    """[summary]

    Arguments:
        LRP {[type]} -- [description]
    """
    def __init__(self, forward_func):
        rules = repeat(BasicRule(), 1000)
        super(LRP_0, self).__init__(forward_func, rules)


class PropagationRule(object):
    '''
    Abstract class to be basis for all propagation rule classes
    '''
    def propagate(self, relevance, layer, activations):
        raise NotImplementedError


class EpsilonRhoRule(PropagationRule):

    def propagate(self, relevance, layer, activations):
        '''
        Implementation of epsilon-rho-rule according to Montavon et al. in Explainable AI book
        Abstract class that serves as basis for other rules.
        Valid for conv and dense layers with ReLU units.
        '''
        activations = (activations.data).requires_grad_(True)
        layer = self._apply_rho(layer)
        z = self._epsilon() + layer.forward(activations)
        s = (relevance/z).data
        (z*s.data).sum().backward()
        c = activations.grad
        propagated_relevance = (activations*c).data

        return propagated_relevance

    def _epsilon(self):
        raise NotImplementedError

    def _rho(self, input):
        raise NotImplementedError

    def _apply_rho(self, layer):
        if layer.weight is not None:
            new_weights = self._rho(layer.weight.data)
            layer.weight.data = new_weights
        if layer.bias is not None:
            new_bias = self._rho(layer.bias.data)
            layer.bias.data = new_bias
        return layer


class BasicRule(EpsilonRhoRule):
    def _epsilon(self):
        return 0

    def _rho(self, input):
        return input