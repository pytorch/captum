#!/usr/bin/env python3
import torch

from .common import (
    _run_forward,
    _format_input_baseline,
    _format_tensor_into_tuples,
    validate_input,
    validate_target,
)
from .gradient import compute_gradients


class Attribution:
    r"""
    All attribution algorithms extend this class. It enforces its child classes
    to extend and override core `attribute` method.
    """

    def attribute(self, inputs, **kwargs):
        r"""
        This method computes and returns the attribution values for each input tensor.
        Deriving classes are responsible for implementing its logic accordingly.

        Args:

            inputs (tensor or tuple of tensors):  Input for which attribution
                        is computed. It can be provided as a single tensor or
                        a tuple of multiple tensors. If mutliple input tensors
                        are provided, the batch sizes must be aligned accross all
                        tensors.
            **kwargs (Any, optional): Arbitrary keyword arguments used by specific
                        attribution algorithms that extend this class.


        Returns:

            attributions (tensor or tuple of tensors): Attribution values for each
                        input vector. The `attributions` have the same shape and
                        dimensionality as the inputs.
                        If a single tensor is provided as inputs, a single tensor
                        is returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        """
        raise NotImplementedError("Deriving class should implement attribute method")

    def has_convergence_delta(self):
        r"""
        This method informs the user whether the attribution algorithm provides
        a convergence delta (aka an approximation error) or not. Convergence
        delta may serve as a proxy of trust in an attribution algorithm. If deriving
        attribution class provides a `compute_convergence_delta` method, it should
        override both `compute_convergence_delta` and `has_convergence_delta` methods.

        Returns:

            has_convergence_delta (bool): Returns whether the attribution algorithm
                        provides a convergence delta (aka approximation error) or not.

        """
        return False

    def compute_convergence_delta(self, attributions, *args):
        r"""
        Deriving attribution algorithms that provide convergence delta (aka
        approximation error) should implement this method. Convergence error can
        be computed based on some properties of the attribution alogrithms.
        For example, integrated gradients, deeplift and gradient SHAP have
        the property that the sum of the attributions must be equal to the difference
        of the model's function at the end and start points of the approximation.

        Args:

                attributions (tensor or tuple of tensors):  Input for which integrated
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
                            are provided, the examples must be aligned appropriately.
                **kwargs (optional):

        Returns:

        """
        raise NotImplementedError(
            "Deriving class should implement" " compute_convergence_delta method"
        )


class GradientAttribution(Attribution):
    r"""
    All gradient based attribution algorithms extend this class. It requires a
    forward function, which most commonly is the forward function of the model
    that we want to interpret or the model itself.
    """

    def __init__(self, forward_func):
        r"""
        Args

            forward_func (callable or torch.nn.Module): This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
        """
        super().__init__()
        self.forward_func = forward_func
        self.gradient_func = compute_gradients

    def compute_convergence_delta(
        self,
        attributions,
        start_point,
        end_point,
        target=None,
        additional_forward_args=None,
    ):
        r"""
        Args:
                attributions (tensor or tuple of tensors):
                start_point (tensor or tuple of tensors, optional):  Baseline define
                            the starting point from which integral is computed.
                            If inputs is a single tensor, baselines must also be a
                            single tensor with exactly the same dimensions as inputs.
                            If inputs is a tuple of tensors, baselines must also be
                            a tuple of tensors, with matching dimensions to inputs.
                            Default: zero tensor for each input tensor
                end_point (tensor or tuple of tensors):  Input for which integrated
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
                            are provided, the examples must be aligned appropriately.
                target (int, optional):  Output index for which gradient is computed
                            (for classification cases, this is the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary. (Note: Tuples for multi
                            -dimensional output indices will be supported soon.)
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
                            correspond to the number of examples.
                            Default: None

        Returns:

        """
        end_point, start_point = _format_input_baseline(end_point, start_point)
        attributions = _format_tensor_into_tuples(attributions)

        num_samples = end_point[0].shape[0]
        validate_input(end_point, start_point)
        validate_target(num_samples, target)

        def _sum_rows(input):
            return input.view(input.shape[0], -1).sum(1)

        with torch.no_grad():
            start_point = _sum_rows(
                _run_forward(
                    self.forward_func, start_point, target, additional_forward_args
                )
            )

            end_point = _sum_rows(
                _run_forward(
                    self.forward_func, end_point, target, additional_forward_args
                )
            )
        row_sums = [_sum_rows(attribution) for attribution in attributions]
        attr_sum = torch.tensor([sum(row_sum) for row_sum in zip(*row_sums)])
        return attr_sum - (end_point - start_point)


class InternalAttribution(GradientAttribution):
    r"""
    Shared base class for LayerAttrubution and NeuronAttribution,
    attribution types that require a model and a particular layer.
    """

    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable or torch.nn.Module):  This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
            layer (torch.nn.Module): Layer for which output attributions are computed.
                        Output size of attribute matches that of layer output.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                        applies a DataParallel model, which allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func)
        self.layer = layer
        self.device_ids = device_ids


class LayerAttribution(InternalAttribution):
    r"""
    Layer attribution provides attribution values for the given layer, quanitfying
    the importance of each neuron within the given layer's output. The output
    attribution of calling attribute on a LayerAttribution object always matches
    the size of the layer output.
    """

    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable or torch.nn.Module):  This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
            layer (torch.nn.Module): Layer for which output attributions are computed.
                        Output size of attribute matches that of layer output.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                        applies a DataParallel model, which allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer, device_ids)


class NeuronAttribution(InternalAttribution):
    r"""
        Neuron attribution provides input attribution for a given neuron, quanitfying
        the importance of each input feature in the activation of a particular neuron.
        Calling attribute on a NeuronAttribution object requires also providing
        the index of the neuron in the output of the given layer for which attributions
        are required.
        The output attribution of calling attribute on a NeuronAttribution object
        always matches the size of the input.
    """

    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable or torch.nn.Module):  This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
            layer (torch.nn.Module): Layer for which output attributions are computed.
                        Output size of attribute matches that of layer output.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                        applies a DataParallel model, which allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(self, inputs, neuron_index, **kwargs):
        r"""
        This method computes and returns the neuron attribution values for each
        input tensor. Deriving classes are responsible for implementing
        its logic accordingly.

        Args:

                inputs:     A single high dimensional input tensor or a tuple of them.
                neuron_index (int or tuple): Tuple providing index of neuron in output
                        of given layer for which attribution is desired. Length of
                        this tuple must be one less than the number of
                        dimensions in the output of the given layer (since
                        dimension 0 corresponds to number of examples).

        Returns:

                attributions (tensor or tuple of tensors): Attribution values for
                        each input vector. The `attributions` have the
                        dimensionality of inputs.
        """
        raise NotImplementedError("A derived class should implement attribute method")
