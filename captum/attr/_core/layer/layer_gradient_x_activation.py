#!/usr/bin/env python3
from typing import Any, Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

from ..._utils.attribution import GradientAttribution, LayerAttribution
from ..._utils.common import (
    _format_additional_forward_args,
    _format_attributions,
    _format_input,
)
from ..._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)


class LayerGradientXActivation(LayerAttribution, GradientAttribution):
    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Optional[List[int]] = None,
    ) -> None:
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
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: Optional[
            Union[int, Tuple[int, ...], Tensor, List[Tuple[int, ...]]]
        ] = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
            Computes element-wise product of gradient and activation for selected
            layer on given inputs.

            Args:

                inputs (tensor or tuple of tensors):  Input for which attributions
                            are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if multiple input tensors
                            are provided, the examples must be aligned appropriately.
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
                additional_forward_args (any, optional): If the forward function
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
                attribute_to_layer_input (bool, optional): Indicates whether to
                            compute the attribution with respect to the layer input
                            or output. If `attribute_to_layer_input` is set to True
                            then the attributions will be computed with respect to
                            layer input, otherwise it will be computed with respect
                            to layer output.
                            Default: False

            Returns:
                *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                            Product of gradient and activation for each
                            neuron in given layer output.
                            Attributions will always be the same size as the
                            output of the given layer.
                            Attributions are returned in a tuple based on whether
                            the layer inputs / outputs are contained in a tuple
                            from a forward hook. For standard modules, inputs of
                            a single tensor are usually wrapped in a tuple, while
                            outputs of a single tensor are not.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx12x32x32.
                >>> net = ImageClassifier()
                >>> layer_ga = LayerGradientXActivation(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes layer activation x gradient for class 3.
                >>> # attribution size matches layer output, Nx12x32x32
                >>> attribution = layer_ga.attribute(input, 3)
        """
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals, is_layer_tuple = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(
            is_layer_tuple,
            tuple(
                layer_gradient * layer_eval
                for layer_gradient, layer_eval in zip(layer_gradients, layer_evals)
            ),
        )
