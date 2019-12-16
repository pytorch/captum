#!/usr/bin/env python3

import torch
import torch.nn as nn


class PropagationRule(object):
    '''
    Abstract class to be basis for all propagation rule classes
    '''
    def propagate(self, relevance, layer, activations):
        raise NotImplementedError


class EpsilonRhoRule(PropagationRule):

    def __init__(self, epsilon, rho):
        self.epsilon = epsilon
        self.rho = rho


    def propagate(self, relevance, layer, activations):
        '''
        Implementation of epsilon-rho-rule according to Montavon et al. in Explainable AI book
        Abstract class that serves as basis for other rules.
        Valid for conv and dense layers with ReLU units.
        '''
        activations = (activations.data).requires_grad_(True)
        layer = self._apply_rho(layer)
        z = self.epsilon + layer.forward(activations)
        s = (relevance/z).data
        (z*s.data).sum().backward()
        c = activations.grad
        propagated_relevance = (activations*c).data

        return propagated_relevance


    def _apply_rho(self, layer):
        if layer.weight is not None:
            new_weights = self.rho(layer.weight.data)
            layer.weight.data = new_weights
        if layer.bias is not None:
            new_bias = self.rho(layer.bias.data)
            layer.bias.data = new_bias
        return layer


class BasicRule(EpsilonRhoRule):
    '''
    Basic rule for relevance propagation, also called LRP-0, see Montavon et al. Explainable AI book.
    Implemented using the EpsilonRhoRule.
    Used for upper layers.
    '''
    def __init__(self):
        super(BasicRule, self).__init__(epsilon=0, rho=lambda x: x)


class EpsilonRule(EpsilonRhoRule):
    '''
    Rule for relevance propagation using a small value of epsilon to avoid numerical instabilities.
    Used for middle layers.
    '''
    def __init__(self, epsilon=1e-9):
        super(EpsilonRule, self).__init__(epsilon, rho=lambda x: x)


class GammaRule(EpsilonRhoRule):
    '''
    Gamma rule for relevance propagation, 
    Used for lower layers.
    '''
    def __init__(self, gamma=2):

        def _gamma_function(tensor_input):
            tensor_positive = torch.clamp(tensor_input, min=0)
            tensor_output = tensor_input + (gamma * tensor_positive)
            return tensor_output

        super(GammaRule, self).__init__(epsilon=0, rho=_gamma_function)


class Alpha1_Beta0_Rule(EpsilonRhoRule):
    '''
    Alpha1_Beta0 rule for relevance backpropagation, also known as Deep-Taylor.
    Used for lower layers.
    '''
    def __init__(self):

        def _rho_function(tensor_input):
            tensor_output = torch.clamp(tensor_input, min=0)
            return tensor_output

        super(Alpha1_Beta0_Rule, self).__init__(epsilon=0, rho=_rho_function)