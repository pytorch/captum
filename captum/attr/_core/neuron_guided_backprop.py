#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.common import format_input, _format_attributions
from .._utils.gradient import (
    _forward_layer_eval_with_neuron_grads,
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from .guided_backprop import GuidedBackprop


class NeuronGuidedBackprop(NeuronAttribution, GuidedBackprop):
    def __init__(self, model, layer, device_ids=None):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                          Attributions for a particular neuron in the output of
                          this layer are computed using the argument neuron_index
                          in the attribute method.
                          Currently, only layers with a single tensor output are
                          supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super(NeuronAttribution, self).__init__(model, layer, device_ids)
        super(GuidedBackprop, self).__init__(model)

    def attribute(self, inputs, neuron_index, additional_forward_args=None):
        r""""
        Computes attribution of the given neuron using guided backpropagation.
        Guided backpropagation computes the gradient of the target neuron
        with respect the input, but gradients of ReLU functions are overriden
        so that only non-negative gradients are backpropagated.

        More details regarding the guided backpropagation algorithm can be found
        in the original paper here:
        https://arxiv.org/abs/1412.6806

        Warning: Ensure that all ReLU operations in the forward function of the
        given model are performed using a module (nn.module.ReLU).
        If nn.functional.ReLU is used, gradients are not overriden appropriately.

        Args:

            inputs (tensor or tuple of tensors):  Input for which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        mutliple input tensors are provided, the examples must
                        be aligned appropriately.
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
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Guided backprop attribution of
                        particular neuron with respect to each input feature.
                        Attributions will always be the same size as the provided
                        inputs, with each value providing the attribution of the
                        corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> neuron_gb = NeuronGuidedBackprop(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # To compute neuron attribution, we need to provide the neuron
            >>> # index for which attribution is desired. Since the layer output
            >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
            >>> # which indexes a particular neuron in the layer output.
            >>> # For this example, we choose the index (4,1,2).
            >>> # Computes neuron guided backpropagation for neuron with
            >>> # index (4,1,2).
            >>> attribution = neuron_gb.attribute(input, (4,1,2))
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # set hooks for overriding ReLU gradients
        self.model.apply(self._register_hooks)

        _, input_grads = _forward_layer_eval_with_neuron_grads(
            self.forward_func,
            inputs,
            self.layer,
            additional_forward_args,
            neuron_index,
            device_ids=self.device_ids,
        )

        # remove set hooks
        self._remove_hooks()

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, input_grads)
