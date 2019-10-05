#!/usr/bin/env python3
import torch

from .common import _run_forward
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
        delta_per_sample=False,
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
        # Currently this is provided when sample_delta is True, but this
        # should change to the general behavior.
        if delta_per_sample:
            return attr_sum - (end_point - start_point)
        else:
            return abs(attr_sum - (end_point - start_point)).sum().item()


class GradientAttribution(Attribution):
    r"""
    All gradient based attribution algorithms extend this class. It requires a
    forward function, which most commonly is the forward function of the model
    that we want to interpret.
    """

    def __init__(self, forward_func):
        r"""
        Args:

            forward_func:  The forward function of the model or any modification of it
        """
        super().__init__()
        self.forward_func = forward_func
        self.gradient_func = compute_gradients


class InternalAttribution(GradientAttribution):
    r"""
    Shared base class for LayerAttrubution and NeuronAttribution,
    attribution types that require a model and a particular layer.
    """

    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

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
        Args:

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
