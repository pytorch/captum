#!/usr/bin/env python3

from itertools import repeat

import torch
import torch.nn as nn

from .._utils.common import _format_attributions, _format_input, _run_forward
from .._utils.attribution import Attribution, GradientAttribution
from .._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    _forward_layer_eval,
)
from .lrp_rules import PropagationRule, BasicRule


class LRP(Attribution):
    def __init__(self, model, rules):
        """
        Args:

            model (callable): The forward function of the model or
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
                raise TypeError(
                    "Please select propagation rules inherited from class PropagationRule"
                )
        self.rules = rules
        super(LRP, self).__init__(model)

    def attribute(
        self,
        inputs,
        target=None,
        return_convergence_delta=False,
        additional_forward_args=None,
    ):
        r"""[summary]

            Args:
                inputs (tensor or tuple of tensors):  Input for which relevance is propagated.
                            If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if multiple input tensors
                            are provided, the examples must be aligned appropriately.
                target (int, tuple, tensor or list, optional):  Output indices for
                            which gradients are computed (for classification cases,
                            this is usually the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary.
                            For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                            integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                            the number of examples in inputs (dim 0). Each integer
                            is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                            elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                            examples in inputs (dim 0), and each tuple containing
                            #output_dims - 1 elements. Each tuple is applied as the
                            target for the corresponding example.

                        Default: None
                additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None

                return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions** or 2-element tuple of **attributions**, **delta**::
            - **attributions** (*tensor* or tuple of *tensors*):
                        The propagated relevance values with respect to each
                        input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            - **delta** (*tensor*, returned if return_convergence_delta=True):

                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in input.

        """
        relevances = list()
        is_inputs_tuple = isinstance(inputs, tuple)
        # inputs = _format_input(inputs)
        output = self._get_activations(inputs, None, additional_forward_args)
        relevance = self._mask_relevance(output, target)
        for layer, rule, activation in zip(
            reversed(self.layers), reversed(self.rules), reversed(self.activations)
        ):
            # Convert Max-Pooling to Average Pooling layer
            if isinstance(layer, torch.nn.MaxPool2d):
                layer = torch.nn.AvgPool2d(2)
            # Propagate relevance for Conv2D and Pooling
            if (
                isinstance(layer, torch.nn.Conv2d)
                or isinstance(layer, torch.nn.AvgPool2d)
                or isinstance(layer, torch.nn.AdaptiveAvgPool2d)
                or isinstance(layer, torch.nn.Linear)
            ):
                relevance = rule.propagate(relevance, layer, activation)
            else:
                pass
            relevances.insert(0, relevance)

        if return_convergence_delta:
            delta = self.compute_convergence_delta(
                relevances[0], inputs, additional_forward_args, target
            )
            return _format_attributions(is_inputs_tuple, (relevances,)), delta
        else:
            return _format_attributions(is_inputs_tuple, (relevances,))

    def has_convergence_delta(self):
        return True

    def compute_convergence_delta(
        self, attributions, inputs, additional_forward_args=None, target=None
    ):
        """
        Here, we use the completeness property of LRP: The relevance is conserved 
        during the propagation through the models' layers. Therefore, the difference 
        between the sum of attribution (relevance) values and model output is taken as 
        the convergence delta. It should be zero for functional attribution. However, 
        when rules with an epsilon value are used for stability reasons, relevance is 
        absorbed during propagation and the convergence delta is non-zero.

        Args:

                attributions (tensor or tuple of tensors): Attribution scores that
                            are precomputed by an attribution algorithm.
                            Attributions can be provided in form of a single tensor
                            or a tuple of those. It is assumed that attribution
                            tensor's dimension 0 corresponds to the number of
                            examples, and if multiple input tensors are provided,
                            the examples must be aligned appropriately.
                inputs (tensor or tuple of tensors). Input for which relevance is propagated.
                            If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if multiple input tensors
                            are provided, the examples must be aligned appropriately.

        Keyword Arguments:
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be either a single additional
                            argument of a Tensor or arbitrary (non-tuple) type or a tuple
                            containing multiple additional arguments including tensors
                            or any arbitrary python types. These arguments are provided to
                            forward_func in order, following the arguments in inputs.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                target (int, tuple, tensor or list, optional):  Output indices for
                            which gradients are computed (for classification cases,
                            this is usually the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary.
                            For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                            integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                            the number of examples in inputs (dim 0). Each integer
                            is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                            elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                            examples in inputs (dim 0), and each tuple containing
                            #output_dims - 1 elements. Each tuple is applied as the
                            target for the corresponding example.

                        Default: None

        Returns:
            *tensor*:
            - **delta** Difference of relevance in output layer and input layer.
        """
        relevance = _run_forward(self.model, inputs, target, additional_forward_args)
        return torch.sum(relevance) - torch.sum(attributions)

    def _get_layers(self, model):
        """
        Get list of children modules of the forward function or model.
        """
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                self._get_layers(layer)
            if list(layer.children()) == []:
                self.layers.append(layer)

    def _get_activations(self, inputs, target=None, additional_forward_args=None):
        self.activations = list()

        hooks = list()
        for layer in self.layers:
            registered_hook = layer.register_forward_hook(self._forward_hook)
            hooks.append(registered_hook)
        # self.relevance = self.model(inputs)
        relevance = _run_forward(self.model, inputs, target, additional_forward_args)

        for registered_hook in hooks:
            registered_hook.remove()

        return relevance

    def _forward_hook(self, module, input, output):
        self.activations.append(*input)

    def _mask_relevance(self, output, target=None):
        """
        If target is class of classification task, the output layer is masked with zeros except the class output.
        """
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
