#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.gradient import construct_neuron_grad_fn
from .guided_backprop_deconvnet import GuidedBackprop, Deconvolution


class NeuronDeconvolution(NeuronAttribution):
    def __init__(self, model, layer, device_ids=None):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
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
        super().__init__(model, layer, device_ids)
        self.deconv = Deconvolution(model)

    def attribute(
        self,
        inputs,
        neuron_index,
        additional_forward_args=None,
        attribute_to_neuron_input=False,
    ):
        r""""
        Computes attribution of the given neuron using deconvolution.
        Deconvolution computes the gradient of the target output with
        respect to the input, but gradients of ReLU functions are overriden so
        that the gradient of the ReLU input is simply computed taking ReLU of
        the output gradient, essentially only propagating non-negative gradients
        (without dependence on the sign of the ReLU input).

        More details regarding the deconvolution algorithm can be found
        in these papers:
        https://arxiv.org/abs/1311.2901
        https://link.springer.com/chapter/10.1007/978-3-319-46466-4_8

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
            attribute_to_neuron_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the neuron input
                        or output. If `attribute_to_neuron_input` is set to True
                        then the attributions will be computed with respect to
                        neuron's inputs, otherwise it will be computed with respect
                        to neuron's outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal neuron, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Deconvolution attribution of
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
            >>> neuron_deconv = NeuronDeconvolution(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # To compute neuron attribution, we need to provide the neuron
            >>> # index for which attribution is desired. Since the layer output
            >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
            >>> # which indexes a particular neuron in the layer output.
            >>> # For this example, we choose the index (4,1,2).
            >>> # Computes neuron deconvolution for neuron with
            >>> # index (4,1,2).
            >>> attribution = neuron_deconv.attribute(input, (4,1,2))
        """
        self.deconv.gradient_func = construct_neuron_grad_fn(
            self.layer, neuron_index, self.device_ids, attribute_to_neuron_input
        )
        return self.deconv.attribute(inputs, None, additional_forward_args)


class NeuronGuidedBackprop(NeuronAttribution):
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
        super().__init__(model, layer, device_ids)
        self.guided_backprop = GuidedBackprop(model)

    def attribute(
        self,
        inputs,
        neuron_index,
        additional_forward_args=None,
        attribute_to_neuron_input=False,
    ):
        r""""
        Computes attribution of the given neuron using guided backpropagation.
        Guided backpropagation computes the gradient of the target neuron
        with respect to the input, but gradients of ReLU functions are overriden
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
        self.guided_backprop.gradient_func = construct_neuron_grad_fn(
            self.layer, neuron_index, self.device_ids, attribute_to_neuron_input
        )
        return self.guided_backprop.attribute(inputs, None, additional_forward_args)
