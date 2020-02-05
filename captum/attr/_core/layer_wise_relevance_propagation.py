#!/usr/bin/env python3

from itertools import repeat
import warnings

import torch
import torch.nn as nn
import torch.onnx
import torch.onnx.utils

from .._utils.common import (
    _format_attributions,
    _format_input,
    _run_forward,
    _select_targets,
)
from .._utils.attribution import Attribution, GradientAttribution
from .._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    compute_gradients,
    _forward_layer_eval,
)
from .._utils.lrp_rules import PropagationRule, Z_Rule


class LRP(Attribution):
    def __init__(self, model, rules):
        """
        Args:

            model (callable): The forward function of the model or
                        any modification of it
            rules (iterator of PropagationRules or None): List of Rules for each layer
                        of forward_func. For layers where the relevances are not propagated
                        (i.e. ReLU) the rule can be None.
        """
        self.model = model
        self.layerset = []
        self._get_layerset(model)
        self.changes_weights = False

        for rule in rules:
            if not isinstance(rule, PropagationRule) and rule is not None:
                raise TypeError(
                    "Please select propagation rules inherited from class PropagationRule"
                )
            if hasattr(rule, "forward_hook_weights"):
                self.changes_weights = True
        self.rules = rules
        super(LRP, self).__init__(model)

    def attribute(
        self,
        inputs,
        target=None,
        return_convergence_delta=False,
        additional_forward_args=None,
        return_for_all_layers=False,
        verbose=False,
    ):
        """
            Layer-wise relevance propagation is based on a backward propagation mechanism applied sequentially
            to all layers of the model. Here, the model output score represents the initial relevance which is
            decomposed into values for each neuron of the underlying layers. The decomposition is defined
            by rules that are chosen for each layer, involving its weights and activations. Details on the model
            can be found in the original paper [https://doi.org/10.1371/journal.pone.0130140] and on the implementation
            and rules in the tutorial paper [https://doi.org/10.1016/j.dsp.2017.10.011].

            Attention: The implementation is only tested for ReLU activation layers, linear and conv2D layers. An error
            is raised if other activation functions (sigmoid, tanh) are used.
            Skip connections as in Resnets are not supported.

            #TODO: To generalize to a broader range of models the detection of layers and their connections
            need to be improved. One way may be to get the graph of the model and evaluate all graph nodes
            to use them for backward propagation in a way similar to the description in [https://arxiv.org/abs/1904.04734]

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

                return_for_all_layers (bool, optional): Indicates whether to return
                        relevance values for all layers. If False, only the relevance
                        values for the input layer are returned.

                verbose (bool, optional): Indicates whether information on skipped layers
                        during propagation is printed.

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
        Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> from lrp_rules import Alpha1_Beta0_Rule
                >>> net = ImageClassifier()
                >>> rules = [Alpha1_Beta0_Rule(), None]
                >>> lrp = LRP(net, rules)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = lrp.attribute(input, target=5)

        """
        self.return_for_all_layers = return_for_all_layers
        self.verbose = verbose
        self.backward_handles = list()
        self.forward_handles = list()
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = _format_input(inputs)
        output = self._get_layers(inputs, None, additional_forward_args)
        gradient_mask = apply_gradient_requirements(inputs)

        self._register_forward_hooks()

        self._change_weights(inputs)

        relevances = compute_gradients(self.model, inputs, target_ind=target)

        output = _select_targets(output, target)
        relevances = tuple(
            relevance * input * output for relevance, input in zip(relevances, inputs)
        )

        if self.return_for_all_layers:
            relevances = [*relevances]
            for layer in self.layers:
                if hasattr(layer, "relevance"):
                    relevances.append(layer.relevance)
                else:
                    relevances.append(relevances[-1])
            relevances = (relevances,)

        self._remove_backward_hooks()
        self._remove_forward_hooks()
        undo_gradient_requirements(inputs, gradient_mask)

        if return_convergence_delta:
            delta = self.compute_convergence_delta(
                relevances[0], inputs, additional_forward_args, target
            )
            return _format_attributions(is_inputs_tuple, relevances), delta
        else:
            return _format_attributions(is_inputs_tuple, relevances)

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

    def _get_layerset(self, model):
        """
        Get list of children modules of the forward function or model.
        Checks wether Sigmoid or Tanh activations are used and raises error if that is the case.
        """
        for layer in model.children():
            if list(layer.children()) == []:
                if isinstance(layer, (nn.Sigmoid, nn.Tanh)):
                    raise TypeError(
                        "Invalid activation used. Implementation is only valid for ReLU activations."
                    )
                self.layerset.append(layer)
            else:
                self._get_layerset(layer)

    def _get_layers(self, inputs, target=None, additional_forward_args=None):
        # TODO: Check if still necessary or get_layerset is sufficient
        self.layers = list()
        hooks = list()

        for layer in self.layerset:
            registered_hook = layer.register_forward_hook(self._forward_hook)
            hooks.append(registered_hook)
        output = _run_forward(self.model, inputs, target, additional_forward_args)

        for registered_hook in hooks:
            registered_hook.remove()

        return output

    def _register_forward_hooks(self):
        for layer, rule in zip(self.layers, self.rules):
            # Convert Max-Pooling to Average Pooling layer
            # TODO: Adapt for max pooling layers, layer in model is not changed for backward pass.
            # if isinstance(layer, torch.nn.MaxPool2d):
            #    layer = torch.nn.AvgPool2d(layer.kernel_size)
            # Propagate relevance for Conv2D, Linear and Pooling
            if isinstance(
                layer,
                (
                    torch.nn.MaxPool2d,
                    torch.nn.Conv2d,
                    torch.nn.AvgPool2d,
                    torch.nn.AdaptiveAvgPool2d,
                    torch.nn.Linear,
                ),
            ):
                if self.verbose:
                    print(f"\nRelevance propagated over {layer} with manipulation.")
                forward_handle = layer.register_forward_hook(rule.forward_hook)
                self.forward_handles.append(forward_handle)
                if self.return_for_all_layers:
                    relevance_handle = layer.register_backward_hook(
                        rule._backward_hook_relevance
                    )
                    self.backward_handles.append(relevance_handle)

            else:
                if self.verbose:
                    print(f"\nRelevance passed over {layer} without manipulation.")
                backward_handle = layer.register_backward_hook(
                    rule.backward_hook_activation
                )
                self.backward_handles.append(backward_handle)

    def _forward_hook(self, module, input, output):
        # setattr(module, 'activations', *input)
        self.layers.append(module)

    def _register_weight_hooks(self):
        for layer, rule in zip(self.layers, self.rules):
            if hasattr(rule, "forward_hook_weights"):
                if isinstance(
                    layer,
                    (
                        torch.nn.MaxPool2d,
                        torch.nn.Conv2d,
                        torch.nn.AvgPool2d,
                        torch.nn.AdaptiveAvgPool2d,
                        torch.nn.Linear,
                    ),
                ):
                    forward_handle = layer.register_forward_hook(
                        rule.forward_hook_weights
                    )
                    self.forward_handles.append(forward_handle)

    def _change_weights(self, inputs):
        if self.changes_weights:
            self._register_weight_hooks()
            _ = _run_forward(self.model, inputs)

    def _remove_forward_hooks(self):
        for forward_handle in self.forward_handles:
            forward_handle.remove()

    def _remove_backward_hooks(self):
        for backward_handle in self.backward_handles:
            backward_handle.remove()
        for rule in self.rules:
            if hasattr(rule, "_handle_input_hook"):
                rule._handle_input_hook.remove()
            if hasattr(rule, "_handle_output_hook"):
                rule._handle_output_hook.remove()
            if hasattr(rule, "_handle_layer_hook"):
                rule._handle_layer_hook.remove()


class LRP_0(LRP):
    """
    LRP class, which uses the base rule for every layer.
    """

    def __init__(self, forward_func):
        rules = repeat(BasicRule(), 1000)
        super(LRP_0, self).__init__(forward_func, rules)
