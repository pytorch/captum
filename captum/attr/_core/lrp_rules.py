#!/usr/bin/env python3

import copy

import torch
import torch.nn as nn

#TODO: Implement w^2 rule
#class w2_Rule(PropagationRule):
#TODO: Documentation


class PropagationRule(object):
    '''
    Abstract class to be basis for all propagation rule classes
    '''
    def propagate(self, relevance, layer, activations):
        raise NotImplementedError


class EpsilonRhoRule(PropagationRule):
    '''
    Example rho function:
        def _rho_function(tensor_input):
            tensor_positive = torch.clamp(tensor_input, min=0)
            tensor_output = tensor_input + (0.25 * tensor_positive)
            return tensor_output
    '''
    def __init__(self, rho=lambda p: p, epsilon=lambda z: z+1e-9):
        self.epsilon = epsilon
        self.rho = rho


    def propagate(self, relevance, layer, activations):
        '''
        Implementation of epsilon-rho-rule according to Montavon et al. in Explainable AI book
        Class that serves as basis for other rules.
        Valid for conv and dense layers with ReLU units.
        '''
        activations = (activations.data).requires_grad_(True)
        layer_copy = copy.deepcopy(layer)
        layer_rho = self._apply_rho(layer_copy)
        z =  self.epsilon(layer_rho.forward(activations))
        # Correct shape when tensor is flattened
        if relevance.shape != z.shape:
            relevance = relevance.view(z.shape)
        s = (relevance/z).data
        (z*s.data).sum().backward()
        c = activations.grad
        propagated_relevance = (activations*c).data

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
    '''
    Basic rule for relevance propagation, also called LRP-0, see Montavon et al. Explainable AI book.
    Implemented using the EpsilonRhoRule.
    Used for upper layers.
    '''
    def __init__(self):
        super(BasicRule, self).__init__()


class EpsilonRule(EpsilonRhoRule):
    '''
    Rule for relevance propagation using a small value of epsilon to avoid numerical instabilities.
    Used for middle layers.
    '''
    def __init__(self, epsilon=lambda z: z+1e-9):
        super(EpsilonRule, self).__init__(epsilon=epsilon)


class GammaRule(EpsilonRhoRule):
    '''
    Gamma rule for relevance propagation,
    Used for lower layers.
    '''
    def __init__(self, gamma=0.25):

        def _gamma_function(tensor_input):
            tensor_positive = torch.clamp(tensor_input, min=0)
            tensor_output = tensor_input + (gamma * tensor_positive)
            return tensor_output

        super(GammaRule, self).__init__(rho=_gamma_function)


class Alpha1_Beta0_Rule(EpsilonRhoRule):
    '''
    Alpha1_Beta0 rule for relevance backpropagation, also known as Deep-Taylor.
    Used for lower layers.
    '''
    def __init__(self):

        def _rho_function(tensor_input):
            tensor_output = torch.clamp(tensor_input, min=0)
            return tensor_output

        super(Alpha1_Beta0_Rule, self).__init__(rho=_rho_function)


class zB_Rule(PropagationRule):
    '''
    For pixel layers
    If image is normalized, the mean and std need to be given
    '''
    def __init__(self, minimum_value, maximum_value, mean=None, std=None):
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.mean = mean
        self.std = std


    def propagate(self, relevance, layer, activations):
        '''
        Implementation of epsilon-rho-rule according to Montavon et al. in Explainable AI book
        Class that serves as basis for other rules.
        Valid for conv and dense layers with ReLU units.
        '''
        stability_factor = 1e-9
        activations = (activations.data).requires_grad_(True)

        if self.mean is not None and self.std is not None:
            mean = torch.Tensor(self.mean).reshape(1,-1,1,1)
            std  = torch.Tensor(self.std).reshape(1,-1,1,1)
            lb = (activations.data*0+(0-mean)/std).requires_grad_(True)
            hb = (activations.data*0+(1-mean)/std).requires_grad_(True)
        else:
            lb = torch.Tensor(activations.size()).fill_(self.minimum_value).requires_grad_(True)
            hb = torch.Tensor(activations.size()).fill_(self.maximum_value).requires_grad_(True)

        layer_copy = copy.deepcopy(layer)
        z =  layer_copy.forward(activations) + stability_factor
        z = z - self._layer_copy_negative_weights(layer).forward(hb)
        z = z - self._layer_copy_positive_weights(layer).forward(lb)
        s = (relevance/z).data
        (z*s.data).sum().backward()
        c, cp, cm = activations.grad, lb.grad, hb.grad
        propagated_relevance = (activations*c + lb*cp + hb*cm).data

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

