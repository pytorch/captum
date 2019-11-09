#!/usr/bin/env python3
import torch
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import LayerAttribution
from .._utils.batching import _batched_operator
from .._utils.common import (
    _reshape_and_sum,
    _format_input_baseline,
    _format_additional_forward_args,
    _expand_additional_forward_args,
    validate_input,
    _expand_target,
)
from .._utils.gradient import compute_layer_gradients_and_eval


class LayerConductance(LayerAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer.
                          Currently, it is assumed that the inputs or the outputs
                          of the layer, depending on which one is used for
                          attribution, can only be a single tensor.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer, device_ids)

    def has_convergence_delta(self):
        return True

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="riemann_trapezoid",
        internal_batch_size=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False,
    ):
        r"""
            Computes conductance with respect to the given layer. The
            returned output is in the shape of the layer's output, showing the total
            conductance of each hidden layer neuron.

            The details of the approach can be found here:
            https://arxiv.org/abs/1805.12233
            https://arxiv.org/pdf/1807.09946.pdf

            Note that this provides the total conductance of each neuron in the
            layer's output. To obtain the breakdown of a neuron's conductance by input
            features, utilize NeuronConductance instead, and provide the target
            neuron index.

            Args:

                inputs (tensor or tuple of tensors):  Input for which layer
                            conductance is computed. If forward_func takes a single
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
                            correspond to the number of examples. It will be repeated
                            for each of `n_steps` along the integrated path.
                            For all other types, the given argument is used for
                            all forward evaluations.
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
                            If internal_batch_size is None, then all evaluations are
                            processed in one batch.
                            Default: None
                return_convergence_delta (bool, optional): Indicates whether to return
                            convergence delta or not. If `return_convergence_delta`
                            is set to True convergence delta will be returned in
                            a tuple following attributions.
                            Default: False
                attribute_to_layer_input (bool, optional): Indicates whether to
                            compute the attribution with respect to the layer input
                            or output. If `attribute_to_layer_input` is set to True
                            then the attributions will be computed with respect to
                            layer inputs, otherwise it will be computed with respect
                            to layer outputs.
                            Note that currently it is assumed that either the input
                            or the output of internal layer, depending on whether we
                            attribute to the input or output, is a single tensor.
                            Support for multiple tensors will be added later.
                            Default: False

            Returns:
                **attributions** or 2-element tuple of **attributions**, **delta**:
                - **attributions** (*tensor*):
                            Conductance of each neuron in given layer input or
                            output. Attributions will always be the same size as
                            the input or output of the given layer, depending on
                            whether we attribute to the inputs or outputs
                            of the layer which is decided by the input flag
                            `attribute_to_layer_input`.
                - **delta** (*tensor*, returned if return_convergence_delta=True):
                            The difference between the total
                            approximated and true conductance.
                            This is computed using the property that the total sum of
                            forward_func(inputs) - forward_func(baselines) must equal
                            the total sum of the attributions.
                            Delta is calculated per example, meaning that the number of
                            elements in returned delta tensor is equal to the number of
                            of examples in inputs.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx12x32x32.
                >>> net = ImageClassifier()
                >>> layer_cond = LayerConductance(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes layer conductance for class 3.
                >>> # attribution size matches layer output, Nx12x32x32
                >>> attribution = layer_cond.attribute(input, target=3)
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        validate_input(inputs, baselines, n_steps, method)

        num_examples = inputs[0].shape[0]

        # Retrieve scaling factors for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        alphas = alphas_func(n_steps + 1)

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
        # dim -> (#examples * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps + 1)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps + 1)

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = _batched_operator(
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

        # Compute differences between consecutive evaluations of layer_eval.
        # This approximates the total input gradient of each step multiplied
        # by the step size.
        grad_diffs = layer_eval[num_examples:] - layer_eval[:-num_examples]

        # Element-wise mutliply gradient of output with respect to hidden layer
        # and summed gradients with respect to input (chain rule) and sum
        # across stepped inputs.
        attributions = _reshape_and_sum(
            grad_diffs * layer_gradients[:-num_examples],
            n_steps,
            num_examples,
            layer_eval.shape[1:],
        )
        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            delta = self.compute_convergence_delta(
                (attributions,),
                start_point,
                end_point,
                target=target,
                additional_forward_args=additional_forward_args,
            )
            return attributions, delta
        return attributions
