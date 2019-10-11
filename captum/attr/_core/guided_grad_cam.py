#!/usr/bin/env python3

from .._utils.attribution import GradientAttribution, LayerAttribution
from .._utils.common import format_input

from .grad_cam import LayerGradCam
from .guided_backprop import GuidedBackprop


class GuidedGradCam(GradientAttribution):
    def __init__(self, model, layer, device_ids=None):
        r"""
        Args

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which GradCAM attributions are computed.
                          Currently, only layers with a single tensor output are
                          supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(model)
        self.grad_cam = LayerGradCam(model, layer, device_ids)
        self.guided_backprop = GuidedBackprop(model)

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        chosen_input_index=0,
        interpolate_mode="nearest",
    ):
        r"""
            Computes element-wise product of guided backpropagation attributions
            with upsampled GradCAM attributions. GradCAM attributions are computed
            with respect to the layer provided in the constructor, and attributions
            are upsampled to match the input size. GradCAM is designed for
            convolutional neural networks, and is usually applied to the last
            convolutional layer.

            Note that if multiple input tensors are provided, only attributions
            for the tensor with index chosen_input_index are returned. This tensor
            should be spatially alligned with the chosen layer for the results
            to be meaningful, e.g. an input image tensor for a convolutional layer.

            In addition, this tensor must have the same number of dimensions as the
            chosen layer's output, and must be either 3D, 4D or 5D.

            More details regarding GuidedGradCAM can be found in the original
            GradCAM paper here:
            https://arxiv.org/pdf/1610.02391.pdf

            Warning: Ensure that all ReLU operations in the forward function of the
            given model are performed using a module (nn.module.ReLU).
            If nn.functional.ReLU is used, gradients are not overriden appropriately.

            Args:

                inputs (tensor or tuple of tensors):  Input for which attributions
                            are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
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
                chosen_input_index (int, optional): If multiple input tensors are
                            provided, this argument defines the tensor for which
                            attributions should be computed. This tensor should be
                            spatially alligned with the given layer for the results
                            to be meaningful, e.g. an input image tensor for a
                            convolutional layer.
                interpolate_mode (str, optional): Method for interpolation, which
                            muat be a valid input interpolation mode for
                            torch.nn.functional. These methods are
                            "nearest", "area", "linear" (3D-only), "bilinear"
                            (4D-only), "bicubic" (4D-only), "trilinear" (5D-only)
                            based on the number of dimensions of the chosen layer
                            output (which must also match the number of
                            dimensions for inputs[chosen_input_index]). Note that
                            the original GradCAM paper uses "bilinear"
                            interpolation, but we default to "nearest" for
                            applicability to any of 3D, 4D or 5D tensors.
                            Default: "nearest"

            Returns:
                *tensor* of **attributions**:
                - **attributions** (*tensor*):
                        Element-wise product of (upsampled) GradCAM
                        and Guided Backprop attributions for tensor with index
                        chosen_input_index.
                        Attributions will be the same size as the input tensor
                        at index chosen_input_index.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv4, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx50x8x8.
                >>> # It is the last convolution layer, which is the recommended
                >>> # use case for GuidedGradCAM.
                >>> net = ImageClassifier()
                >>> guided_gc = GuidedGradCam(net, net.conv4)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes guided GradCAM attributions for class 3.
                >>> # attribution size matches input size, Nx3x32x32
                >>> attribution = guided_gc.attribute(input, 3)
        """
        inputs = format_input(inputs)
        grad_cam_attr = self.grad_cam.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        guided_backprop_attr = self.guided_backprop.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        return guided_backprop_attr[chosen_input_index] * LayerAttribution.interpolate(
            grad_cam_attr,
            inputs[chosen_input_index].shape[2:],
            interpolate_mode=interpolate_mode,
        )
