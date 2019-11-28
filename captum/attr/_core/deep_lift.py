#!/usr/bin/env python3
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .._utils.common import (
    _format_input,
    _format_baseline,
    _format_callable_baseline,
    _format_attributions,
    _format_tensor_into_tuples,
    _run_forward,
    _validate_input,
    _expand_target,
    _expand_additional_forward_args,
    _tensorize_baseline,
    _call_custom_attribution_func,
    _compute_conv_delta_and_format_attrs,
    ExpansionTypes,
)
from .._utils.attribution import GradientAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


# Check if module backward hook can safely be used for the module that produced
# this inputs / outputs mapping
def _check_valid_module(inputs, outputs):
    curr_fn = outputs.grad_fn
    first_next = curr_fn.next_functions[0]
    try:
        return first_next[0] == inputs[first_next[1]].grad_fn
    except IndexError:
        return False


class DeepLift(GradientAttribution):
    def __init__(self, model):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
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
        self.forward_handles = []
        self.forward_handles_refs = []
        self.backward_handles = []

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        return_convergence_delta=False,
        custom_attribution_func=None,
    ):
        r""""
        Implements DeepLIFT algorithm based on the following paper:
        Learning Important Features Through Propagating Activation Differences,
        Avanti Shrikumar, et. al.
        https://arxiv.org/abs/1704.02685

        and the gradient formulation proposed in:
        Towards better understanding of gradient-based attribution methods for
        deep neural networks,  Marco Ancona, et.al.
        https://openreview.net/pdf?id=Sy21R9JAW

        This implementation supports only Rescale rule. RevealCancel rule will
        be supported in later releases.
        In addition to that, in order to keep the implementation cleaner, DeepLIFT
        for internal neurons and layers extends current implementation and is
        implemented separately in LayerDeepLift and NeuronDeepLift.
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

            inputs (tensor or tuple of tensors):  Input for which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                            exactly the same dimensions as inputs or the first
                            dimension is one and the remaining dimensions match
                            with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                            be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                            to each tensor in the inputs' tuple can be:
                            - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding
                                input tensor.
                            - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted
                                for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

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
            custom_attribution_func (callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:
                            - custom_attribution_func(multipliers)
                            - custom_attribution_func(multipliers, inputs)
                            - custom_attribution_func(multipliers, inputs, baselines)
                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.

                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Attribution score computed based on DeepLift rescale rule with respect
                to each input feature. Attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                This is computed using the property that
                the total sum of forward_func(inputs) - forward_func(baselines)
                must equal the total sum of the attributions computed
                based on DeepLift's rescale rule.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                of examples in input.
                Note that the logic described for deltas is guaranteed when the
                default logic for attribution computations is used, meaning that the
                `custom_attribution_func=None`, otherwise it is not guaranteed and
                depends on the specifics of the `custom_attribution_func`.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> dl = DeepLift(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for class 3.
            >>> attribution = dl.attribute(input, target=3)
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_input(inputs)
        baselines = _format_baseline(baselines, inputs)

        gradient_mask = apply_gradient_requirements(inputs)

        _validate_input(inputs, baselines)

        # set hooks for baselines
        warnings.warn(
            """Setting forward, backward hooks and attributes on non-linear
               activations. The hooks and attributes will be removed
            after the attribution is finished"""
        )
        self.model.apply(self._register_hooks_ref)

        baselines = _tensorize_baseline(inputs, baselines)

        _run_forward(
            self.model,
            baselines,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        # remove forward hook set for baselines
        for forward_handles_ref in self.forward_handles_refs:
            forward_handles_ref.remove()

        self.model.apply(self._register_hooks)
        gradients = self.gradient_func(
            self.model,
            inputs,
            target_ind=target,
            additional_forward_args=additional_forward_args,
        )

        if custom_attribution_func is None:
            attributions = tuple(
                (input - baseline) * gradient
                for input, baseline, gradient in zip(inputs, baselines, gradients)
            )
        else:
            attributions = _call_custom_attribution_func(
                custom_attribution_func, gradients, inputs, baselines
            )

        # remove hooks from all activations
        self._remove_hooks()

        undo_gradient_requirements(inputs, gradient_mask)
        return _compute_conv_delta_and_format_attrs(
            self,
            return_convergence_delta,
            attributions,
            baselines,
            inputs,
            additional_forward_args,
            target,
            is_inputs_tuple,
        )

    def _is_non_linear(self, module):
        return type(module) in SUPPORTED_NON_LINEAR.keys()

    # we need forward hook to access and detach the inputs and outputs of a neuron
    def _forward_hook(self, module, inputs, outputs):
        input_attr_name = "input"
        output_attr_name = "output"
        self._detach_tensors(input_attr_name, output_attr_name, module, inputs, outputs)
        if not _check_valid_module(inputs, outputs):
            module.is_invalid = True
            module.saved_grad = None

            def tensor_backward_hook(grad):
                if module.saved_grad is None:
                    raise RuntimeError(
                        """Module {} was detected as not supporting correctly module
                        backward hook. You should modify your hook to ignore the given
                        grad_inputs (recompute them by hand if needed) and save the
                        newly computed grad_inputs in module.saved_grad. See MaxPool1d
                        as an example.""".format(
                            module
                        )
                    )
                return module.saved_grad

            inputs[0].register_hook(tensor_backward_hook)
        else:
            module.is_invalid = False

    def _forward_hook_ref(self, module, inputs, outputs):
        input_attr_name = "input_ref"
        output_attr_name = "output_ref"
        self._detach_tensors(input_attr_name, output_attr_name, module, inputs, outputs)

    def _detach_tensors(
        self, input_attr_name, output_attr_name, module, inputs, outputs
    ):
        inputs = _format_tensor_into_tuples(inputs)
        outputs = _format_tensor_into_tuples(outputs)
        setattr(module, input_attr_name, tuple(input.detach() for input in inputs))
        setattr(module, output_attr_name, tuple(output.detach() for output in outputs))

    def _backward_hook(self, module, grad_input, grad_output, eps=1e-10):
        r"""
         `grad_input` is the gradient of the neuron with respect to its input
         `grad_output` is the gradient of the neuron with respect to its output
          we can override `grad_input` according to chain rule with.
         `grad_output` * delta_out / delta_in.

         """
        delta_in = tuple(
            inp - inp_ref for inp, inp_ref in zip(module.input, module.input_ref)
        )
        delta_out = tuple(
            out - out_ref for out, out_ref in zip(module.output, module.output_ref)
        )
        multipliers = tuple(
            SUPPORTED_NON_LINEAR[type(module)](
                module, delta_in, delta_out, list(grad_input), grad_output, eps=eps
            )
        )
        # remove all the properies that we set for the inputs and output
        del module.input_ref
        del module.output_ref
        del module.input
        del module.output

        return multipliers

    def _can_register_hook(self, module):
        # TODO find a better way of checking if a module is a container or not
        module_fullname = str(type(module))
        has_already_hooks = len(module._backward_hooks) > 0
        return not (
            "nn.modules.container" in module_fullname
            or has_already_hooks
            or not self._is_non_linear(module)
        )

    def _register_hooks_ref(self, module):
        if not self._can_register_hook(module):
            return
        forward_handle_ref = module.register_forward_hook(self._forward_hook_ref)
        self.forward_handles_refs.append(forward_handle_ref)

    def _register_hooks(self, module):
        if not self._can_register_hook(module):
            return
        # adds forward hook to leaf nodes that are non-linear
        forward_handle = module.register_forward_hook(self._forward_hook)
        backward_handle = module.register_backward_hook(self._backward_hook)
        self.forward_handles.append(forward_handle)
        self.backward_handles.append(backward_handle)

    def _remove_hooks(self):
        for forward_handle in self.forward_handles:
            forward_handle.remove()
        for backward_handle in self.backward_handles:
            backward_handle.remove()

    def has_convergence_delta(self):
        return True


class DeepLiftShap(DeepLift):
    def __init__(self, model):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
        """
        DeepLift.__init__(self, model)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        return_convergence_delta=False,
        custom_attribution_func=None,
    ):
        r"""
        Extends DeepLift algorithm and approximates SHAP values using Deeplift.
        For each input sample it computes DeepLift attribution with respect to
        each baseline and averages resulting attributions.
        More details about the algorithm can be found here:

        http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

        Note that the explanation model:
            1. Assumes that input features are independent of one another
            2. Is linear, meaning that the explanations are modeled through
               the additive composition of feature effects.
        Although, it assumes a linear model for each explanation, the overall
        model across multiple explanations can be complex and non-linear.

        Args:

            inputs (tensor or tuple of tensors):  Input for which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, callable, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                            exactly the same dimensions as inputs or the first
                            dimension is one and the remaining dimensions match
                            with inputs.

                        - a tuple of tensors, the baseline corresponding
                            to each tensor in the inputs' tuple is either a tensor
                            with matching dimensions to corresponding tensor in
                            the inputs' tuple or the first dimension is one and the
                            remaining dimensions match with the corresponding input
                            tensor.

                        - callable function, optionally takes `inputs` as an
                            argument and either returns a single tensor
                            or a tuple of those.
                        The number of samples in the baselines' tensors must be
                        larger than one.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

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
            custom_attribution_func (callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:
                            - custom_attribution_func(multipliers)
                            - custom_attribution_func(multipliers, inputs)
                            - custom_attribution_func(multipliers, inputs, baselines)
                        In case this function is not provided we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attribution score computed based on DeepLift rescale rule with
                        respect to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
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
                        Note that the logic described for deltas is guaranteed
                        when the default logic for attribution computations is used,
                        meaning that the `custom_attribution_func=None`, otherwise
                        it is not guaranteed and depends on the specifics of the
                        `custom_attribution_func`.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> dl = DeepLiftShap(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes shap values using deeplift for class 3.
            >>> attribution = dl.attribute(input, target=3)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_input(inputs)
        baselines = _format_callable_baseline(baselines, inputs)

        assert isinstance(baselines[0], torch.Tensor) and baselines[0].shape[0] > 1, (
            "Baselines distribution has to be provided in form of a torch.Tensor"
            " with more than one example but found: {}."
            " If baselines are provided in shape of scalars or with a single"
            " baseline example, `DeepLift`"
            " approach can be used instead.".format(baselines[0])
        )

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        (
            exp_inp,
            exp_base,
            exp_tgt,
            exp_addit_args,
        ) = self._expand_inputs_baselines_targets(
            baselines, inputs, target, additional_forward_args
        )

        attributions = super().attribute(
            exp_inp,
            exp_base,
            target=exp_tgt,
            additional_forward_args=exp_addit_args,
            return_convergence_delta=return_convergence_delta,
            custom_attribution_func=custom_attribution_func,
        )
        if return_convergence_delta:
            attributions, delta = attributions

        attributions = tuple(
            self._compute_mean_across_baselines(inp_bsz, base_bsz, attribution)
            for attribution in attributions
        )

        if return_convergence_delta:
            return _format_attributions(is_inputs_tuple, attributions), delta
        else:
            return _format_attributions(is_inputs_tuple, attributions)

    def _expand_inputs_baselines_targets(
        self, baselines, inputs, target, additional_forward_args
    ):
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        expanded_inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input in inputs
            ]
        )
        expanded_baselines = tuple(
            [
                baseline.repeat(
                    (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
                ).requires_grad_()
                for baseline in baselines
            ]
        )
        expanded_target = _expand_target(
            target, base_bsz, expansion_type=ExpansionTypes.repeat_interleave
        )
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, base_bsz)
            if additional_forward_args is not None
            else None
        )
        return (
            expanded_inputs,
            expanded_baselines,
            expanded_target,
            input_additional_args,
        )

    def _compute_mean_across_baselines(self, inp_bsz, base_bsz, attribution):
        # Average for multiple references
        attr_shape = (inp_bsz, base_bsz)
        if len(attribution.shape) > 1:
            attr_shape += attribution.shape[1:]
        return torch.mean(attribution.view(attr_shape), axis=1, keepdim=False)


def nonlinear(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    r"""
    grad_input: (dLoss / dprev_layer_out, dLoss / wij, dLoss / bij)
    grad_output: (dLoss / dlayer_out)
    https://github.com/pytorch/pytorch/issues/12331
    """
    # supported non-linear modules take only single tensor as input hence accessing
    # only the first element in `grad_input` and `grad_output`
    grad_input[0] = torch.where(
        abs(delta_in[0]) < eps,
        grad_input[0],
        grad_output[0] * delta_out[0] / delta_in[0],
    )
    return grad_input


def softmax(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    grad_input_unnorm = torch.where(
        abs(delta_in[0]) < eps,
        grad_input[0],
        grad_output[0] * delta_out[0] / delta_in[0],
    )
    # normalizing
    n = np.prod(grad_input[0].shape)
    grad_input[0] = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return grad_input


def maxpool1d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module,
        F.max_pool1d,
        F.max_unpool1d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool2d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module,
        F.max_pool2d,
        F.max_unpool2d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool3d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module,
        F.max_pool3d,
        F.max_unpool3d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool(
    module,
    pool_func,
    unpool_func,
    delta_in,
    delta_out,
    grad_input,
    grad_output,
    eps=1e-10,
):
    # The forward function of maxpool takes only tensors not
    # a tuple hence accessing the first
    # element in the tuple of inputs, grad_input and grad_output
    _, indices = pool_func(
        module.input[0],
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        module.ceil_mode,
        True,
    )
    unpool_grad_out_delta = unpool_func(
        grad_output[0] * delta_out[0],
        indices,
        module.kernel_size,
        module.stride,
        module.padding,
        list(module.input[0].shape),
    )

    # If the module is invalid, we need to recompute the grad_input
    if module.is_invalid:
        original_grad_input = grad_input
        grad_input = (
            unpool_func(
                grad_output[0],
                indices,
                module.kernel_size,
                module.stride,
                module.padding,
                list(module.input[0].shape),
            ),
        )

    new_grad_input = torch.where(
        abs(delta_in[0]) < eps, grad_input[0], unpool_grad_out_delta / delta_in[0]
    )

    # If the module is invalid, save the newly computed gradients
    # The original_grad_input will be overridden later in the Tensor hook
    if module.is_invalid:
        module.saved_grad = new_grad_input
        return original_grad_input
    else:
        return (new_grad_input,)


SUPPORTED_NON_LINEAR = {
    nn.ReLU: nonlinear,
    nn.ELU: nonlinear,
    nn.LeakyReLU: nonlinear,
    nn.Sigmoid: nonlinear,
    nn.Tanh: nonlinear,
    nn.Softplus: nonlinear,
    nn.MaxPool1d: maxpool1d,
    nn.MaxPool2d: maxpool2d,
    nn.MaxPool3d: maxpool3d,
    nn.Softmax: softmax,
}
