#!/usr/bin/env python3

from itertools import repeat

import torch
import torch.nn as nn

from .._utils.common import _format_attributions, _format_input, _run_forward
from .._utils.attribution import Attribution, GradientAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements, _forward_layer_eval
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
        self.model = model
        self.layers = self._get_layers(model)
        self._layer_count = len(self.layers)
        for rule in rules:
            if not isinstance(rule, PropagationRule):
                raise TypeError("Please select propagation rules inherited from class PropagationRule")
        self.rules = rules
        super(LRP, self).__init__(model)


    def attribute(self, inputs, target, **kwargs):
        '''
        target: index of target class for which relevance is computed
        '''
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = _format_input(inputs)
        #TODO: general, handle flatten layer in models.
        activations = self._get_activations(inputs)
        #TODO: apply mask to relevance to show only relevant class.
        relevance_tensor = activations[-1]
        relevance = _run_forward(self.model, inputs, target=target)
        for layer, rule, activation in zip(reversed(self.layers), reversed(self.rules), reversed(activations[:-1])):
            # Convert Max-Pooling to Average Pooling layer
            if isinstance(layer, torch.nn.MaxPool2d):
                layer = torch.nn.AvgPool2d(2)
            # linear to conv layer
            """ if isinstance(layer, torch.nn.Linear):
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = nn.Parameter(layer.bias)
            layer = newlayer """
            # Propagate relevance for Conv2D and Pooling
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.AvgPool2d):
                relevance = rule.propagate(relevance, layer, activation)
            else:
                pass
        
        return _format_attributions(is_inputs_tuple, relevance)



    def _get_layers(self, model):
        '''
        Get list of children modules of the forward function or model.
        '''
        all_layers = []
        return self._remove_sequential(model, all_layers)


    def _remove_sequential(self, model, all_layers):

        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                self._remove_sequential(layer, all_layers)
            if list(layer.children()) == []:
                all_layers.append(layer)
        return all_layers


    def _get_activations(self, inputs):
        '''
        Calculate activations for inputs for every layer of forward_func.
        '''
        #TODO: use _forward_layer_eval
        activations = [inputs] + [None] * (self._layer_count)
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):# and isinstance(layers[index-1], nn.poo
                activations[index + 1] = layer.forward(activations[index].view(-1, layer.in_features))
            else:
                activations[index + 1] = layer.forward(activations[index])
        return activations


    def _get_activations_hook(self, inputs):
        self.activations = list()
        self.z_values = list()
        for layer in self.layers:
            layer.register_forward_hook(self._forward_hook)
        out = self.model(inputs)


    def _forward_hook(self, module, input, output):
        self.activations.append(input)
        self.z_values.append(output)

class LRP_0(LRP):
    """LRP class, which uses the base rule for every layer.

    Arguments:
        LRP {[type]} -- [description]
    """
    def __init__(self, forward_func):
        rules = repeat(BasicRule(), 1000)
        super(LRP_0, self).__init__(forward_func, rules)

