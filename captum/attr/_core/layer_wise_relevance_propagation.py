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
        self.layers = []
        self._get_layers(model)
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
        relevances = list()
        is_inputs_tuple = isinstance(inputs, tuple)
        #inputs = _format_input(inputs)
        self._get_activations(inputs)
        relevance = self._mask_relevance(self.relevance, target)
        for layer, rule, activation in zip(reversed(self.layers), reversed(self.rules), reversed(self.activations)):#[:-1]
            # Convert Max-Pooling to Average Pooling layer
            if isinstance(layer, torch.nn.MaxPool2d):
                layer = torch.nn.AvgPool2d(2)
            # Propagate relevance for Conv2D and Pooling
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.AvgPool2d) or isinstance(layer, torch.nn.AdaptiveAvgPool2d) or isinstance(layer, torch.nn.Linear):
                relevance = rule.propagate(relevance, layer, activation)
            else:
                pass
            print('relevance: ' + str(torch.sum(relevance)))
            relevances.insert(0, relevance)
        return _format_attributions(is_inputs_tuple, (relevances, ))


    def _get_layers(self, model):
        '''
        Get list of children modules of the forward function or model.
        '''
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                self._get_layers(layer)
            if list(layer.children()) == []:
                self.layers.append(layer)


    def _get_activations(self, inputs):
        self.activations = list()

        hooks = list()
        for layer in self.layers:
            registered_hook = layer.register_forward_hook(self._forward_hook)
            hooks.append(registered_hook)
        self.relevance = self.model(inputs)
        for registered_hook in hooks:
            registered_hook.remove()


    def _forward_hook(self, module, input, output):
        self.activations.append(*input)


    def _mask_relevance(self, output, target=None):
        '''
        If target is class of classification task, the output layer is masked with zeros except the class output.
        '''
        if target is None:
            return output
        elif isinstance(target, int):
            masked_output = torch.zeros(output.size())
            masked_output[slice(None), target] = output[slice(None), target]
            return masked_output


class LRP_0(LRP):
    """LRP class, which uses the base rule for every layer.

    Arguments:
        LRP {[type]} -- [description]
    """
    def __init__(self, forward_func):
        rules = repeat(BasicRule(), 1000)
        super(LRP_0, self).__init__(forward_func, rules)

