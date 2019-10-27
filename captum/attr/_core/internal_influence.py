#!/usr/bin/env python3
import torch
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import LayerAttribution
from .._utils.batching import _batched_operator
from .._utils.common import (
    _reshape_and_sum,
    _format_input_baseline,
    validate_input,
    _format_additional_forward_args,
    _expand_additional_forward_args,
    _expand_target,
)
from .._utils.gradient import compute_layer_gradients_and_eval


class InternalInfluence(LayerAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's output
                          dimensions, corresponding to attribution of each neuron
                          in the output of this layer.
                          Currently, only layers with a single tensor output are
                          supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        internal_batch_size=None,
        attribute_to_layer_input=False,
    ):
        r"""
            Computes internal influence by approximating the integral of gradients
            for a particular layer along the path from a baseline input to the
            given input.
            If no baseline is provided, the default baseline is the zero tensor.
            More details on this approach can be found here:
            https://arxiv.org/pdf/1802.03788.pdf

            Note that this method is similar to applying integrated gradients and
            taking the layer as input, integrating the gradient of the layer with
            respect to the output.

            Args:

                inputs (tensor or tuple of tensors):  Input for which internal
                            influence is computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if multiple input tensors
                            are provided, the examples must be aligned appropriately.
                baselines (tensor or tuple of tensors, optional):  Baseline from which
                            integral is computed. If inputs is a single tensor,
                            baselines must also be a single tensor with exactly the same
                            dimensions as inputs. If inputs is a tuple of tensors,
                            baselines must also be a tuple of tensors, with matching
                            dimensions to inputs.
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
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            For a tensor, the first dimension of the tensor must
                            correspond to the number of examples. It will be
                            repeated for each of `n_steps` along the integrated
                            path. For all other types, the given argument is used
                            for all forward evaluations.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                n_steps (int, optional): The number of steps used by the approximation
                            method. Default: 50.
                method (string, optional): Method for approximating the integral,
                            one of `riemann_right`, `riemann_left`, `riemann_middle`,
                            `riemann_trapezoid` or `gausslegendre`.
                            Default: `gausslegendre` if no method is provided.
                internal_batch_size (int, optional): Divides total #steps * #examples
                            data points into chunks of size internal_batch_size,
                            which are computed (forward / backward passes)
                            sequentially.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain internal_batch_size / num_devices examples.
                            If internal_batch_size is None, then all evaluations
                            are processed in one batch.
                            Default: None
                attribute_to_layer_input (bool, optional): Indicates whether to
                            compute the attribution with respect to layer input
                            or output. If `attribute_to_layer_input` is set to True
                            then the attributions will be computed with respect to
                            layer inputs otherwise it will be computed with respect
                            to layer outputs.
                            Default: False

            Returns:
                *tensor* of **attributions**:
                - **attributions** (*tensor*):
                            Internal influence of each neuron in given
                            layer output. Attributions will always be the same size
                            as the output or input of the given layer depending on
                            whether `attribute_to_layer_input` is set to `False` or
                            `True`respectively.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx12x32x32.
                >>> net = ImageClassifier()
                >>> layer_int_inf = InternalInfluence(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes layer internal influence.
                >>> # attribution size matches layer output, Nx12x32x32
                >>> attribution = layer_int_inf.attribute(input)
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        validate_input(inputs, baselines, n_steps, method)

        # Retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # Compute scaled inputs from baseline to final input.
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguemnts
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # Returns gradient of output with respect to hidden layer.
        layer_gradients, _ = _batched_operator(
            compute_layer_gradients_and_eval,
            scaled_features_tpl,
            input_additional_args,
            internal_batch_size=internal_batch_size,
            forward_fn=self.forward_func,
            layer=self.layer,
            target_ind=expanded_target,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        # flattening grads so that we can multipy it with step-size
        # calling contigous to avoid `memory whole` problems
        scaled_grads = layer_gradients.contiguous().view(n_steps, -1) * torch.tensor(
            step_sizes
        ).view(n_steps, 1).to(layer_gradients.device)

        # aggregates across all steps for each tensor in the input tuple
        return _reshape_and_sum(
            scaled_grads, n_steps, inputs[0].shape[0], layer_gradients.shape[1:]
        )
