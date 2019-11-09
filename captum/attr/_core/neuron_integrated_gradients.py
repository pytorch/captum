#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.gradient import construct_neuron_grad_fn

from .integrated_gradients import IntegratedGradients


class NeuronIntegratedGradients(NeuronAttribution):
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
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer, device_ids)

    def attribute(
        self,
        inputs,
        neuron_index,
        baselines=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        internal_batch_size=None,
        attribute_to_neuron_input=False,
    ):
        r"""
            Approximates the integral of gradients for a particular neuron
            along the path from a baseline input to the given input.
            If no baseline is provided, the default baseline is the zero tensor.
            More details regarding the integrated gradient method can be found in the
            original paper here:
            https://arxiv.org/abs/1703.01365

            Note that this method is equivalent to applying integrated gradients
            where the output is the output of the identified neuron.


            Args:

                inputs (tensor or tuple of tensors):  Input for which neuron integrated
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
                            can be provided. It must be either a single additional
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            For a tensor, the first dimension of the tensor must
                            correspond to the number of examples. It will be
                            repeated for each of `n_steps` along the integrated
                            path. For all other types, the given argument is used
                            for all forward evaluations.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                n_steps (int, optional): The number of steps used by the approximation
                            method. Default: 50.
                method (string, optional): Method for approximating the integral,
                            one of `riemann_right`, `riemann_left`, `riemann_middle`,
                            `riemann_trapezoid` or `gausslegendre`.
                            Default: `gausslegendre` if no method is provided.
                internal_batch_size (int, optional): Divides total #steps * #examples
                            data points into chunks of size internal_batch_size,
                            which are computed (forward / backward passes)
                            sequentially.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain internal_batch_size / num_devices examples.
                            If internal_batch_size is None, then all evaluations are
                            processed in one batch.
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
                            Integrated gradients for particular neuron with
                            respect to each input feature.
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
                >>> # To compute neuron attribution, we need to provide the neuron
                >>> # index for which attribution is desired. Since the layer output
                >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
                >>> # which indexes a particular neuron in the layer output.
                >>> # For this example, we choose the index (4,1,2).
                >>> # Computes neuron integrated gradients for neuron with
                >>> # index (4,1,2).
                >>> attribution = neuron_ig.attribute(input, (4,1,2))
        """
        ig = IntegratedGradients(self.forward_func)
        ig.gradient_func = construct_neuron_grad_fn(
            self.layer, neuron_index, self.device_ids, attribute_to_neuron_input
        )
        # Return only attributions and not delta
        return ig.attribute(
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
        )
