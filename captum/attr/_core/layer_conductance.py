#!/usr/bin/env python3
import torch
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import LayerAttribution
from .._utils.common import (
    _reshape_and_sum,
    _format_input_baseline,
    _format_additional_forward_args,
    _expand_additional_forward_args,
    validate_input,
)
from .._utils.gradient import compute_layer_gradients_and_eval


class LayerConductance(LayerAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's output,
                          corresponding to attribution of each neuron in the
                          output of this layer.
                          Currently, only layers which output a single tensor
                          are supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="riemann_trapezoid",
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

            Args

                inputs (tensor or tuple of tensors):  Input for which layer
                            conductance is computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
                            are provided, the examples must be aligned appropriately.
                baselines (tensor or tuple of tensors, optional):  Baseline from which
                            integral is computed. If inputs is a single tensor,
                            baselines must also be a single tensor with exactly the same
                            dimensions as inputs. If inputs is a tuple of tensors,
                            baselines must also be a tuple of tensors, with matching
                            dimensions to inputs.
                            Default: zero tensor for each input tensor
                target (int, optional):  Output index for which gradient is computed
                            (for classification cases, this is the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary. (Note: Tuples for multi
                            -dimensional output indices will be supported soon.)
                            Default: None
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be a tuple containing tensors or
                            any arbitrary python types. These arguments are provided to
                            forward_func in order following the arguments in inputs.
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
                batch_size (int, optional): Divides total #steps * #examples of
                            necessary forward and backward evaluations into chunks
                            of size batch_size, which are evaluated sequentially.
                            If batch_size is None, then all evaluations are processed
                            in one batch.
                            Default: None

            Return

                attributions (tensor): Conductance of each neuron in given layer output.
                            Attributions will always be the same size as the
                            output of the given layer.

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

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            scaled_features_tpl,
            target,
            input_additional_args,
        )
        # Compute differences between consecutive evaluations of layer_eval.
        # This approximates the total input gradient of each step multiplied
        # by the step size.
        grad_diffs = layer_eval[num_examples:] - layer_eval[:-num_examples]

        # Element-wise mutliply gradient of output with respect to hidden layer
        # and summed gradients with respect to input (chain rule) and sum
        # across stepped inputs.
        return _reshape_and_sum(
            grad_diffs * layer_gradients[:-num_examples],
            n_steps,
            num_examples,
            layer_eval.shape[1:],
        )
