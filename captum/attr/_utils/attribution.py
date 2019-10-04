#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from .common import zeros, _run_forward
from .gradient import compute_gradients


class Attribution:
    def attribute(self, inputs, **kwargs):
        r"""
        This method computes and returns the attribution values for each input tensor
        Deriving classes are responsible for implementing its logic accordingly.

        Args:

                inputs:     A single high dimensional input tensor or a tuple of them.

        Returns:

                attributions: Attribution values for each input vector. The
                              `attributions` have the dimensionality of inputs
                              for standard attribution derived classes and the
                              dimensionality of the given tensor for layer attributions.
                others ?

        """
        raise NotImplementedError("A derived class should implement attribute method")

    def _has_convergence_delta(self):
        return False

    def _compute_convergence_delta(
        self,
        attributions,
        start_point,
        end_point,
        target=None,
        additional_forward_args=None,
        is_multi_baseline=False,
    ):
        def _sum_rows(input):
            return torch.tensor([input_row.sum() for input_row in input])

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
        # TODO ideally do not sum - we should return deltas as a 1D tensor
        # of batch size. Let the user to sum it if they need to
        # Address this in a separate PR
        if is_multi_baseline:
            return abs(attr_sum - (end_point - start_point.mean(0).item())).sum().item()
        else:
            return abs(attr_sum - (end_point - start_point)).sum().item()


class GradientAttribution(Attribution):
    def __init__(self, forward_func):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
        """
        super().__init__()
        self.forward_func = forward_func
        self.gradient_func = compute_gradients

    def zero_baseline(self, inputs):
        r"""
        Takes a tuple of tensors as input and returns a tuple that has the same
        size as the `inputs` which contains zero tensors of the same
        shape as the `inputs`

        """
        return zeros(inputs)


class InternalAttribution(GradientAttribution):
    r"""
    Shared base class for LayerAttrubution and NeuronAttribution,
    attribution types that require a model and a particular layer.
    """

    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
            device_ids: Device ID list, necessary only if forward_func applies a
                        DataParallel model, which allows reconstruction of
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
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
            device_ids: Device ID list, necessary only if forward_func applies a
                   DataParallel model, which allows reconstruction of
                   intermediate outputs from batched results across devices.
                   If forward_func is given as the DataParallel model itself,
                   then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer)

    def interpolate(layer_attribution, interpolate_dims):
        return F.interpolate(layer_attribution, interpolate_dims)


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
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
            device_ids: Device ID list, necessary only if forward_func applies a
                   DataParallel model, which allows reconstruction of
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
                neuron_index: Tuple providing index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).

        Returns:

                attributions: Attribution values for each input vector. The
                              `attributions` have the dimensionality of inputs.

        """
        raise NotImplementedError("A derived class should implement attribute method")
