#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from .._utils.attribution import LayerAttribution
from .._utils.common import format_input, _format_additional_forward_args
from .._utils.gradient import compute_layer_gradients_and_eval


class LayerGradCam(LayerAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's output
                          dimensions, corresponding to attribution of each neuron
                          in the output of this layer.
                          Currently, only layers with a single tensor output are
                          supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(self, inputs, target=None, inp_interpolate=False, additional_forward_args=None):
        r"""
            Computes element-wise product of gradient and activation for selected
            layer on given inputs.

            https://arxiv.org/pdf/1610.02391.pdf

            Args

                inputs (tensor or tuple of tensors):  Input for which attributions
                            are computed. If forward_func takes a single
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
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None

            Return

                attributions (tensor): Product of gradient and activation for each
                            neuron in given layer output.
                            Attributions will always be the same size as the
                            output of the given layer.

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
        inputs = format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
        )
        summed_grads = torch.mean(layer_gradients, dim=tuple(x for x in range(2,len(layer_gradients.shape))), keepdim=True)
        print(summed_grads)
        non_neg_scaled_act = F.relu(torch.sum(summed_grads * layer_eval, dim=1, keepdim=True))

        if inp_interpolate:
            return F.interpolate(non_neg_scaled_act, inputs[0].shape[2:])
        return non_neg_scaled_act
