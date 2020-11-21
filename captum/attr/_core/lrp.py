#!/usr/bin/env python3

import warnings

import torch
import torch.nn as nn

from ..._utils.common import _format_input, _format_output, _run_forward
from ..._utils.gradient import apply_gradient_requirements, undo_gradient_requirements
from .._utils.attribution import GradientAttribution
from .._utils.custom_modules import Addition_Module
from .._utils.lrp_rules import EpsilonRule, PropagationRule


class LRP(GradientAttribution):
    r"""
    Layer-wise relevance propagation is based on a backward propagation
    mechanism applied sequentially to all layers of the model. Here, the
    model output score represents the initial relevance which is decomposed
    into values for each neuron of the underlying layers. The decomposition
    is defined by rules that are chosen for each layer, involving its weights
    and activations. Details on the model can be found in the original paper
    [https://doi.org/10.1371/journal.pone.0130140]. The implementation is
    inspired by the tutorial of the same group
    [https://doi.org/10.1016/j.dsp.2017.10.011] and the publication by
    Ancona et al. [https://openreview.net/forum?id=Sy21R9JAW].
    """

    def __init__(self, model) -> None:
        """
        Args:

            model (callable): The forward function of the model or any modification of
                it. Custom rules for a given layer need to be defined as attribute
                `module.rule` and need to be of type PropagationRule. If no rule is
                specified for a layer, a pre-defined default rule for the module type
                is used.
        """
        GradientAttribution.__init__(self, model)

        if isinstance(model, nn.DataParallel):
            warnings.warn(
                """Although input model is of type `nn.DataParallel` it will run
                only on one device. Support for multiple devices will be added soon."""
            )
            self.model = model.module
        else:
            self.model = model

        self._check_rules()

    @property
    def multiplies_by_inputs(self):
        return True

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        return_convergence_delta=False,
        verbose=False,
    ):
        r"""
        Args:
            inputs (tensor or tuple of tensors):  Input for which relevance is
                        propagated. If forward_func takes a single
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

            verbose (bool, optional): Indicates whether information on application
                    of rules is printed during propagation.

        Returns:
            *tensor* or tuple of *tensors* of **attributions**
            or 2-element tuple of **attributions**, **delta**::
            - **attributions** (*tensor* or tuple of *tensors*):
                        The propagated relevance values with respect to each
                        input feature. The values are normalized by the output score
                        value (sum(relevance)=1). To obtain values comparable to other
                        methods or implementations these values need to be multiplied
                        by the output score. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned. The sum of attributions
                        is one and not corresponding to the prediction score as in other
                        implementations.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in the inputs.
        Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> net = ImageClassifier()
                >>> lrp = LRP(net)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = lrp.attribute(input, target=5)

        """
        self.verbose = verbose
        self._original_state_dict = self.model.state_dict()
        self.layers = []
        self._get_layers(self.model)
        self._check_and_attach_rules()
        self.backward_handles = []
        self.forward_handles = []

        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        try:
            # 1. Forward pass: Change weights of layers according to selected rules.
            output = self._compute_output_and_change_weights(
                inputs, target, additional_forward_args
            )
            # 2. Forward pass + backward pass: Register hooks to configure relevance
            # propagation and execute back-propagation.
            self._register_forward_hooks()
            normalized_relevances = self.gradient_func(
                self._forward_fn_wrapper, inputs, target, additional_forward_args
            )
            relevances = tuple(
                normalized_relevance * output.unsqueeze(dim=1)
                for normalized_relevance in normalized_relevances
            )
        finally:
            self._restore_model()

        undo_gradient_requirements(inputs, gradient_mask)

        if return_convergence_delta:
            delta = []
            for relevance in relevances:
                delta.append(self.compute_convergence_delta(relevance, output))
            return (
                _format_output(is_inputs_tuple, relevances),
                _format_output(is_inputs_tuple, tuple(delta)),
            )
        else:
            return _format_output(is_inputs_tuple, relevances)

    def has_convergence_delta(self):
        return True

    def compute_convergence_delta(self, attributions, output):
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

            output (tensor with single element): The output value with respect to which
                        the attribution values are computed. This value corresponds to
                        the target score of a classification model.

        Returns:
            *tensor*:
            - **delta** Difference of relevance in output layer and input layer.
        """

        def _attribution_delta(attributions, output):
            remaining_dims = tuple(range(1, len(attributions.shape)))
            sum_attributions = torch.sum(attributions, dim=remaining_dims)
            delta = output - sum_attributions
            return delta

        if isinstance(attributions, tuple):
            stacked_attributions = torch.stack(attributions, dim=1)
            delta = _attribution_delta(stacked_attributions, output)
        else:
            delta = _attribution_delta(attributions, output)
        return delta

    def _get_layers(self, model):
        for layer in model.children():
            if len(list(layer.children())) == 0:
                self.layers.append(layer)
            else:
                self._get_layers(layer)

    def _check_and_attach_rules(self):
        for layer in self.layers:
            if hasattr(layer, "rule"):
                pass
            elif type(layer) in SUPPORTED_LAYERS_WITH_RULES.keys():
                layer.rule = SUPPORTED_LAYERS_WITH_RULES[type(layer)]()
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None
            else:
                raise TypeError(
                    (
                        f"Module type {type(layer)} is not supported."
                        "No default rule defined."
                    )
                )

    def _check_rules(self):
        for module in self.model.modules():
            if hasattr(module, "rule"):
                if (
                    not isinstance(module.rule, PropagationRule)
                    and module.rule is not None
                ):
                    raise TypeError(
                        (
                            f"Please select propagation rules inherited from class "
                            f"PropagationRule for module: {module}"
                        )
                    )

    def _register_forward_hooks(self):
        for layer in self.layers:
            if type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                backward_handle = layer.register_backward_hook(
                    PropagationRule.backward_hook_activation
                )
                self.backward_handles.append(backward_handle)
            else:
                forward_handle = layer.register_forward_hook(layer.rule.forward_hook)
                self.forward_handles.append(forward_handle)
                if self.verbose:
                    print(f"Applied {layer.rule} on layer {layer}")

    def _register_weight_hooks(self):
        for layer in self.layers:
            if layer.rule is not None:
                forward_handle = layer.register_forward_hook(
                    layer.rule.forward_hook_weights
                )
                self.forward_handles.append(forward_handle)

    def _register_pre_hooks(self):
        for layer in self.layers:
            if layer.rule is not None:
                forward_handle = layer.register_forward_pre_hook(
                    layer.rule.forward_pre_hook_activations
                )
                self.forward_handles.append(forward_handle)

    def _compute_output_and_change_weights(
        self, inputs, target, additional_forward_args
    ):
        try:
            self._register_weight_hooks()
            output = _run_forward(self.model, inputs, target, additional_forward_args)
        finally:
            self._remove_forward_hooks()
        # Register pre_hooks that pass the initial activations from before weight
        # adjustments as inputs to the layers with adjusted weights. This procedure
        # is important for graph generation in the 2nd forward pass.
        self._register_pre_hooks()
        return output

    def _remove_forward_hooks(self):
        for forward_handle in self.forward_handles:
            forward_handle.remove()

    def _remove_backward_hooks(self):
        for backward_handle in self.backward_handles:
            backward_handle.remove()
        for layer in self.layers:
            if hasattr(layer.rule, "_handle_input_hooks"):
                for handle in layer.rule._handle_input_hooks:
                    handle.remove()
            if hasattr(layer.rule, "_handle_output_hook"):
                layer.rule._handle_output_hook.remove()

    def _remove_rules(self):
        for layer in self.layers:
            if hasattr(layer, "rule"):
                del layer.rule

    def _clear_properties(self):
        for layer in self.layers:
            if hasattr(layer, "activation"):
                del layer.activation

    def _restore_state(self):
        self.model.load_state_dict(self._original_state_dict)

    def _restore_model(self):
        self._restore_state()
        self._remove_backward_hooks()
        self._remove_forward_hooks()
        self._remove_rules()
        self._clear_properties()

    def _forward_fn_wrapper(self, *inputs):
        """
        Wraps a forward function with addition of zero as a workaround to
        https://github.com/pytorch/pytorch/issues/35802 discussed in
        https://github.com/pytorch/captum/issues/143#issuecomment-611750044

        #TODO: Remove when bugs are fixed
        """
        adjusted_inputs = tuple(input + 0 for input in inputs)
        return self.model(*adjusted_inputs)


SUPPORTED_LAYERS_WITH_RULES = {
    nn.MaxPool1d: EpsilonRule,
    nn.MaxPool2d: EpsilonRule,
    nn.MaxPool3d: EpsilonRule,
    nn.Conv2d: EpsilonRule,
    nn.AvgPool2d: EpsilonRule,
    nn.AdaptiveAvgPool2d: EpsilonRule,
    nn.Linear: EpsilonRule,
    nn.BatchNorm2d: EpsilonRule,
    Addition_Module: EpsilonRule,
}

SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh]
