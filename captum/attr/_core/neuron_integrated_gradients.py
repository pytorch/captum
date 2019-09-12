#!/usr/bin/env python3
import torch
from .._utils.attribution import NeuronAttribution
from .._utils.common import _forward_layer_eval, _extend_index_list

from .integrated_gradients import IntegratedGradients


class NeuronIntegratedGradients(NeuronAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Neuron index in the attribute method refers to a particular
                          neuron in the output of this layer. Currently, only
                          layers which output a single tensor are supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer)

    def attribute(
        self,
        inputs,
        neuron_index,
        baselines=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
    ):
        r"""
            Approximates the integral of gradients for a particular neuron
            along the path from a baseline input to the given input.
            If no baseline is provided, the default baseline is the zero tensor.
            More details regarding the integrated gradient method can be found in the
            original paper here:
            https://arxiv.org/abs/1703.01365

            Note that this method is equivalent to applying integrated gradients
            taking the output to be the identified neuron.

            Args:

                inputs (tensor or tuple of tensors):  Input for which neuron integrated
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
                            are provided, the examples must be aligned appropriately.
                neuron_index (int or tuple): Index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).
                              An integer may be provided instead of a tuple of
                              length 1.
                baselines (tensor or tuple of tensors, optional):  Baseline from which
                            integral is computed. If inputs is a single tensor,
                            baselines must also be a single tensor with exactly the same
                            dimensions as inputs. If inputs is a tuple of tensors,
                            baselines must also be a tuple of tensors, with matching
                            dimensions to inputs.
                            Default: zero tensor for each input tensor
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be a tuple containing tensors or
                            any arbitrary python types. These arguments are provided to
                            forward_func in order following the arguments in inputs.
                            For a tensor, the first dimension of the tensor must
                            correspond to the number of examples. It will be repeated
                             for each of `n_steps` along the integrated path.
                            For all other types, the given argument is used for
                            all forward evaluations.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                n_steps (int, optional): The number of steps used by the approximation
                            method. Default: 50.
                method (string, optional): Method for approximating the integral,
                            one of `riemann_right`, `riemann_left`, `riemann_middle`,
                            `riemann_trapezoid` or `gausslegendre`.
                            Default: `gausslegendre` if no method is provided.
                batch_size (int, optional): Divides total #steps * #examples of
                            necessary forward and backward evaluations into chunks
                            of size batch_size, which are evaluated sequentially.
                            If batch_size is None, then all evaluations are processed
                            in one batch.
                            Default: None

            Return:

                attributions (tensor or tuple of tensors): Integrated gradients for
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
                >>> neuron_ig = NeuronIntegratedGradients(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes neuron integrated gradients for neuron with
                >>> # index (4,1,2).
                >>> attribution = neuron_ig.attribute(input, (4,1,2))
        """

        def forward_fn(*args):
            layer_output = _forward_layer_eval(self.forward_func, args, self.layer)
            indices = _extend_index_list(args[0].shape[0], neuron_index)
            return torch.stack(tuple(layer_output[i] for i in indices))

        ig = IntegratedGradients(forward_fn)
        # Return only attributions and not delta
        return ig.attribute(
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
        )[0]
