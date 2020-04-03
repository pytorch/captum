#!/usr/bin/env python3
import warnings

import torch
import torch.nn as nn

from ..._core.lrp import LRP
from ..._utils.attribution import LayerAttribution
from ..._utils.common import _format_attributions, _format_input
from ..._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    undo_gradient_requirements,
    _forward_layer_eval,
)


class LayerLRP(LRP, LayerAttribution):
    def __init__(self, model, layer):
        """
        Args:

            model (callable): The forward function of the model or
                        any modification of it. Custom rules for a given layer need to be defined as attribute
                        `module.rule` and need to be of type PropagationRule.
            layer (torch.nn.Module, None): Layer for which attributions are computed.
                          The size and dimensionality of the attributions
                          corresponds to the size and dimensionality of the layer's
                          input or output depending on whether we attribute to the
                          inputs or outputs of the layer. If value is None, the relevance for all layers is returned in attribution.
        """
        LayerAttribution.__init__(self, model, layer)
        LRP.__init__(self, model)

        self._check_rules()

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False,
        verbose=False,
    ):
        """
            Layer-wise relevance propagation is based on a backward propagation mechanism applied sequentially
            to all layers of the model. Here, the model output score represents the initial relevance which is
            decomposed into values for each neuron of the underlying layers. The decomposition is defined
            by rules that are chosen for each layer, involving its weights and activations. Details on the model
            can be found in the original paper [https://doi.org/10.1371/journal.pone.0130140] and on the implementation
            and rules in the tutorial paper [https://doi.org/10.1016/j.dsp.2017.10.011].

            Warning: In-place relus lead to unexptected failures in layer LRP.

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

                verbose (bool, optional): Indicates whether information on application
                        of rules is printed during propagation.

        Returns:
            *tensor* or tuple of *tensors* of **attributions** or 2-element tuple of **attributions**, **delta**::
            - **attributions** (*tensor* or tuple of *tensors*):
                        The propagated relevance values with respect to each
                        input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned. The sum of attributions
                        is one and not corresponding to the prediction score as in other
                        implementations.
            - **delta** (*tensor* or list of *tensors* returned if return_convergence_delta=True):

                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in input.
                        If attributions for all layers are returned (layer=None) a list
                        of tensors is returned with entries for each layer.
        Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> net = ImageClassifier()
                >>> lrp = LRP(net, net.conv1)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = lrp.attribute(input, target=5)

        """
        self.verbose = verbose
        self._original_state_dict = self.model.state_dict()
        self.layers = []
        self._get_layers(self.model)
        self._get_rules()
        self.attribute_to_layer_input = attribute_to_layer_input
        self.backward_handles = []
        self.forward_handles = []

        is_layer_tuple = self._is_layer_tuple(inputs, additional_forward_args)
        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)
        # 1. Forward pass
        self._change_weights(inputs, additional_forward_args)
        self._register_forward_hooks()
        # 2. Forward pass + backward pass
        relevance_input_layer = compute_gradients(
            self.model, inputs, target, additional_forward_args
        )
        # backward hook on input tensor is not called (https://github.com/pytorch/pytorch/issues/35802) 
        self.layers[0].rule.relevance_input = relevance_input_layer[0]
        relevances = self._get_output_relevance()

        self._restore_model()
        undo_gradient_requirements(inputs, gradient_mask)

        if return_convergence_delta:
            if self.layer is None:
                delta = []
                for relevance_layer in relevances[0]:
                    delta.append(self.compute_convergence_delta(relevance_layer))
            else:
                delta = self.compute_convergence_delta(relevances[0])
            return _format_attributions(is_layer_tuple, relevances), delta
        else:
            return _format_attributions(is_layer_tuple, relevances)

    def _is_layer_tuple(self, inputs, additional_forward_args):
        if self.layer is None:
            is_layer_tuple = isinstance(inputs, tuple)
        else:
            with torch.no_grad():
                _, is_layer_tuple = _forward_layer_eval(
                    self.model,
                    inputs,
                    self.layer,
                    additional_forward_args,
                    attribute_to_layer_input=self.attribute_to_layer_input,
                )
        return is_layer_tuple

    def _get_output_relevance(self):
        if self.layer is None:
            relevances = []
            relevance_layers = [
                layer for layer in self.layers if layer.rule is not None
            ]
            for layer in relevance_layers:
                if self.attribute_to_layer_input:
                    relevance = layer.rule.relevance_input
                else:
                    relevance = layer.rule.relevance_output
                relevances.append(relevance)
        else:
            if self.attribute_to_layer_input:
                relevances = self.layer.rule.relevance_input
            else:
                relevances = self.layer.rule.relevance_output

        return (relevances,)
