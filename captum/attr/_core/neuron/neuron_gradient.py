#!/usr/bin/env python3
from ..._utils.attribution import NeuronAttribution, GradientAttribution
from ..._utils.common import (
    _format_input,
    _format_additional_forward_args,
    _format_attributions,
)
from ..._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
    _forward_layer_eval_with_neuron_grads,
)


class NeuronGradient(NeuronAttribution, GradientAttribution):
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
                          then it is not necessary to provide this argument.
        """
        NeuronAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    def attribute(
        self,
        inputs,
        neuron_index,
        additional_forward_args=None,
        attribute_to_neuron_input=False,
    ):
        r"""
            Computes the gradient of the output of a particular neuron with
            respect to the inputs of the network.

            Args:

                inputs (tensor or tuple of tensors):  Input for which neuron
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if multiple input tensors
                            are provided, the examples must be aligned appropriately.
                neuron_index (int or tuple): Index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).
                              An integer may be provided instead of a tuple of
                              length 1.
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be either a single additional
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                attribute_to_neuron_input (bool, optional): Indicates whether to
                            compute the attributions with respect to the neuron input
                            or output. If `attribute_to_neuron_input` is set to True
                            then the attributions will be computed with respect to
                            neuron's inputs, otherwise it will be computed with respect
                            to neuron's outputs.
                            Note that currently it is assumed that either the input
                            or the output of internal neurons, depending on whether we
                            attribute to the input or output, is a single tensor.
                            Support for multiple tensors will be added later.
                            Default: False

            Returns:
                *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                            Gradients of particular neuron with respect to each input
                            feature. Attributions will always be the same size as the
                            provided inputs, with each value providing the attribution
                            of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx12x32x32.
                >>> net = ImageClassifier()
                >>> neuron_ig = NeuronGradient(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # To compute neuron attribution, we need to provide the neuron
                >>> # index for which attribution is desired. Since the layer output
                >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
                >>> # which indexes a particular neuron in the layer output.
                >>> # For this example, we choose the index (4,1,2).
                >>> # Computes neuron gradient for neuron with
                >>> # index (4,1,2).
                >>> attribution = neuron_ig.attribute(input, (4,1,2))
        """
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)

        _, input_grads = _forward_layer_eval_with_neuron_grads(
            self.forward_func,
            inputs,
            self.layer,
            additional_forward_args,
            neuron_index,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_neuron_input,
        )

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, input_grads)
