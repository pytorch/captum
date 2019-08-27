#!/usr/bin/env python3

from .common import zeros
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


class GradientBasedAttribution(Attribution):
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


class InternalAttribution(Attribution):
    r"""
    Shared base class for LayerAttrubution and NeuronAttribution,
    attribution types that require a model and a particular layer.
    """

    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__()
        self.forward_func = forward_func
        self.layer = layer


class LayerAttribution(InternalAttribution):
    r"""
    Layer attribution provides attribution values for the given layer, quanitfying
    the importance of each neuron within the given layer's output. The output
    attribution of calling attribute on a LayerAttribution object always matches
    the size of the layer output.
    """


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
                others ?

        """
        raise NotImplementedError("A derived class should implement attribute method")
