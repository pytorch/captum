#!/usr/bin/env python3

import torch.nn as nn
from .._utils.attribution import LayerAttribution
from .._core.deep_lift import DeepLift, DeepLiftShap
from .._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    _forward_layer_eval,
    compute_layer_gradients_and_eval,
)

from .._utils.common import (
    format_input,
    format_baseline,
    _format_callable_baseline,
    validate_input,
)


class LayerDeepLift(LayerAttribution, DeepLift):
    def __init__(self, model, layer):
        r"""
        Args:

            model (torch.nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which attributions are computed.
                          The size and dimensionality of the attributions
                          corresponds to the size and dimensionality of the layer's
                          input or output depending on whether we attribute to the
                          inputs or outputs of the layer.
                          Currently, it is assumed that both inputs and ouputs of
                          the layer can only be a single tensor.
        """
        if isinstance(model, nn.DataParallel):
            model = model.module

        super(LayerAttribution, self).__init__(model, layer)
        super(DeepLift, self).__init__(model)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False,
    ):
        r""""
        Implements DeepLIFT algorithm for the layer based on the following paper:
        Learning Important Features Through Propagating Activation Differences,
        Avanti Shrikumar, et. al.
        https://arxiv.org/abs/1704.02685

        and the gradient formulation proposed in:
        Towards better understanding of gradient-based attribution methods for
        deep neural networks,  Marco Ancona, et.al.
        https://openreview.net/pdf?id=Sy21R9JAW

        This implementation supports only Rescale rule. RevealCancel rule will
        be supported in later releases.
        Although DeepLIFT's(Rescale Rule) attribution quality is comparable with
        Integrated Gradients, it runs significantly faster than Integrated
        Gradients and is preferred for large datasets.

        Currently we only support a limited number of non-linear activations
        but the plan is to expand the list in the future.

        Note: As we know, currently we cannot access the building blocks,
        of PyTorch's built-in LSTM, RNNs and GRUs such as Tanh and Sigmoid.
        Nonetheless, it is possible to build custom LSTMs, RNNS and GRUs
        with performance similar to built-in ones using TorchScript.
        More details on how to build custom RNNs can be found here:
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Args:

            inputs (tensor or tuple of tensors):  Input for which layer
                        attributions are computed. If forward_func takes a
                        single tensor as input, a single input tensor should be
                        provided. If forward_func takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            baselines (tensor or tuple of tensors, optional): Baselines define
                        reference samples which are compared with the inputs.
                        In order to assign attribution scores DeepLift computes
                        the differences between the inputs and references and
                        corresponding outputs.
                        If inputs is a single tensor, baselines must also be a
                        single tensor with exactly the same dimensions as inputs.
                        If inputs is a tuple of tensors, baselines must also be
                        a tuple of tensors, with matching dimensions to inputs.
                        Default: zero tensor for each input tensor
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
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer inputs, otherwise it will be computed with respect
                        to layer outputs.
                        Note that currently it assumes that both the inputs and
                        outputs of internal layers are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Attribution score computed based on DeepLift's rescale rule with
                respect to layer's inputs or outputs. Attributions will always be the
                same size as the provided layer's inputs or outputs, depending on
                whether we attribute to the inputs or outputs of the layer.
                Since currently it is assumed that the inputs and outputs of the
                layer must be single tensor, returned attributions have the shape
                of that tensor.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                This is computed using the property that the total sum of
                forward_func(inputs) - forward_func(baselines) must equal the
                total sum of the attributions computed based on Deeplift's
                rescale rule.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                of examples in input.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = LayerDeepLift(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and class 3.
            >>> attribution = dl.attribute(input, target=1)
        """
        inputs = format_input(inputs)
        baselines = format_baseline(baselines, inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        validate_input(inputs, baselines)

        # set hooks for baselines
        self.model.apply(self._register_hooks)

        attr_baselines = _forward_layer_eval(
            self.model,
            baselines,
            self.layer,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        # Fixme later: we need to do this because `_forward_layer_eval`
        # always returns a tensor
        attr_baselines = (attr_baselines,)

        # remove forward hook set for baselines
        for forward_handles_ref in self.forward_handles_refs:
            forward_handles_ref.remove()

        gradients, attr_inputs = compute_layer_gradients_and_eval(
            self.model,
            self.layer,
            inputs,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        # Fixme later: we need to do this because
        # `compute_layer_gradients_and_eval` always returns a tensor
        attr_inputs = (attr_inputs,)
        gradients = (gradients,)

        attributions = tuple(
            (input - baseline) * gradient
            for input, baseline, gradient in zip(attr_inputs, attr_baselines, gradients)
        )

        # remove hooks from all activations
        self._remove_hooks()

        undo_gradient_requirements(inputs, gradient_mask)

        return self._compute_conv_delta_and_format_attrs(
            return_convergence_delta,
            attributions,
            baselines,
            inputs,
            additional_forward_args,
            target,
            False, # currently both the input and output of layer can only be a tensor
        )


class LayerDeepLiftShap(LayerDeepLift, DeepLiftShap):
    def __init__(self, model, layer):
        r"""
        Args:

            model (torch.nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which attributions are computed.
                          The size and dimensionality of the attributions
                          corresponds to the size and dimensionality of the layer's
                          input or output depending on whether we attribute to the
                          inputs or outputs of the layer.
                          Currently, it is assumed that both inputs and ouputs of
                          the layer can only be a single tensor.
        """
        super(DeepLiftShap, self).__init__(model)
        super(LayerDeepLift, self).__init__(model, layer)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False,
    ):
        r"""
        Extends LayerDeepLift and DeepLiftShap alogrithms and approximates SHAP
        values for given input `layer`.
        For each input sample - baseline pair it computes DeepLift attributions
        with respect to inputs or outputs of given `layer` averages
        resulting attributions across baselines. Whether to compute the attributions
        with respect to the inputs or outputs of the layer is defined by the
        input flag `attribute_to_layer_input`.
        More details about the algorithm can be found here:

        http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

        Note that the explanation model:
            1. Assumes that input features are independent of one another
            2. Is linear, meaning that the explanations are modeled through
               the additive composition of feature effects.
        Although, it assumes a linear model for each explanation, the overall
        model across multiple explanations can be complex and non-linear.

        Args:

            inputs (tensor or tuple of tensors):  Input for which layer
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (tensor, tuple of tensors or callable, optional): Baselines
                        define reference samples which are compared with the inputs.
                        In order to assign attribution scores DeepLift computes
                        the differences between the inputs and references and
                        corresponding outputs.
                        `baselines` can be either a single tensor, a tuple of
                        tensors or a callable function that, optionally takes
                        `inputs` as an argument and either returns a single tensor
                        or a tuple of those.
                        If inputs is a single tensor, baselines must also be either
                        a single tensor or a function that returns a single tensor.
                        If inputs is a tuple of tensors, baselines must also be
                        either a tuple of tensors or a function that returns a
                        tuple of tensors.
                        The first dimension in baseline tensors defines the
                        distribution of baselines.
                        All other dimensions starting after the first dimension
                        should match with the inputs' dimensions after the
                        first dimension. It is recommended that the number of
                        samples in the baselines' tensors is larger than one.
                        Default: zero tensor for each input tensor
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
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer inputs, otherwise it will be computed with respect
                        to layer outputs.
                        Note that currently it assumes that both the inputs and
                        outputs of internal layers are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attribution score computed based on DeepLift's rescale rule
                        with respect to layer's inputs or outputs. Attributions
                        will always be the same size as the provided layer's inputs
                        or outputs, depending on whether we attribute to the inputs
                        or outputs of the layer. Since currently it is assumed that
                        the inputs and outputs of the layer must be single tensor,
                        returned attributions have the shape of that tensor.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                        This is computed using the property that the
                        total sum of forward_func(inputs) - forward_func(baselines)
                        must be very close to the total sum of attributions
                        computed based on approximated SHAP values using
                        Deeplift's rescale rule.
                        Delta is calculated for each example input and baseline pair,
                        meaning that the number of elements in returned delta tensor
                        is equal to the
                        `number of examples in input` * `number of examples
                        in baseline`. The deltas are ordered in the first place by
                        input example, followed by the baseline.
        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = LayerDeepLiftShap(net, net.conv4)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes shap values using deeplift for class 3.
            >>> attribution = dl.attribute(input, target=3)
        """
        inputs = format_input(inputs)
        baselines = _format_callable_baseline(baselines, inputs)

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        exp_inp, exp_base, exp_target = DeepLiftShap._expand_inputs_baselines_targets(
            self, base_bsz, inp_bsz, baselines, inputs, target
        )
        attributions = LayerDeepLift.attribute(
            self,
            exp_inp,
            exp_base,
            target=exp_target,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        if return_convergence_delta:
            attributions, delta = attributions

        attributions = DeepLiftShap._compute_mean_across_baselines(
            self, inp_bsz, base_bsz, attributions
        )
        if return_convergence_delta:
            return attributions, delta
        else:
            return attributions
