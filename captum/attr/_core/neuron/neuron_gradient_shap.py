#!/usr/bin/env python3

from ..gradient_shap import GradientShap
from ..._utils.attribution import NeuronAttribution, GradientAttribution
from ..._utils.gradient import construct_neuron_grad_fn


class NeuronGradientShap(NeuronAttribution, GradientAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                          The output size of the attribute method matches the
                          dimensions of the inputs or ouputs of the neuron with
                          index `neuron_index` in this layer, depending on whether
                          we attribute to the inputs or outputs of the neuron.
                          Currently, it is assumed that the inputs or the outputs
                          of the neurons in this layer, depending on which one is
                          used for attribution, can only be a single tensor.
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
        baselines=None,
        n_samples=5,
        stdevs=0.0,
        additional_forward_args=None,
        attribute_to_neuron_input=False,
    ):
        r"""
        Implements gradient SHAP for a neuron in a hidden layer based on the
        implementation from SHAP's primary author. For reference, please, view:

        https://github.com/slundberg/shap\
        #deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models

        A Unified Approach to Interpreting Model Predictions
        http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

        GradientShap approximates SHAP values by computing the expectations of
        gradients by randomly sampling from the distribution of baselines/references.
        It adds white noise to each input sample `n_samples` times, selects a
        random baseline from baselines' distribution and a random point along the
        path between the baseline and the input, and computes the gradient of the
        neuron with index `neuron_index` with respect to those selected random
        points. The final SHAP values represent the expected values of
        `gradients * (inputs - baselines)`.

        GradientShap makes an assumption that the input features are independent
        and that the explanation model is linear, meaning that the explanations
        are modeled through the additive composition of feature effects.
        Under those assumptions, SHAP value can be approximated as the expectation
        of gradients that are computed for randomly generated `n_samples` input
        samples after adding gaussian noise `n_samples` times to each input for
        different baselines/references.

        In some sense it can be viewed as an approximation of integrated gradients
        by computing the expectations of gradients for different baselines.

        Current implementation uses Smoothgrad from `NoiseTunnel` in order to
        randomly draw samples from the distribution of baselines, add noise to input
        samples and compute the expectation (smoothgrad).

        Args:

            inputs (tensor or tuple of tensors):  Input for which SHAP attribution
                        values are computed. If `forward_func` takes a single
                        tensor as input, a single input tensor should be provided.
                        If `forward_func` takes multiple tensors as input, a tuple
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
            baselines (scalar, tensor, tuple of scalars or tensors, callable, optional):
                        Baselines define the starting point from which expectation
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                            exactly the same dimensions as inputs or the first
                            dimension is one and the remaining dimensions match
                            with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                            be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                            to each tensor in the inputs' tuple can be:
                            - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding
                                input tensor.
                            - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted
                                for corresponding input tensor.

                        - callable function, optionally takes `inputs` as an
                            argument and either returns a single tensor
                            or a tuple of those.

                        It is recommended that the number of samples in the baselines'
                        tensors is larger than one.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            n_samples (int, optional):  The number of randomly generated examples
                        per sample in the input batch. Random examples are
                        generated by adding gaussian random noise to each sample.
                        Default: `5` if `n_samples` is not provided.
            stdevs    (float, or a tuple of floats optional): The standard deviation
                        of gaussian noise with zero mean that is added to each
                        input in the batch. If `stdevs` is a single float value
                        then that same value is used for all inputs. If it is
                        a tuple, then it must have the same length as the inputs
                        tuple. In this case, each stdev value in the stdevs tuple
                        corresponds to the input with the same index in the inputs
                        tuple.
                        Default: 0.0
            additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It can contain a tuple of ND tensors or
                        any arbitrary python type of any shape.
                        In case of the ND tensor the first dimension of the
                        tensor must correspond to the batch size. It will be
                        repeated for each `n_steps` for each randomly generated
                        input sample.
                        Note that the gradients are not computed with respect
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
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attribution score computed based on GradientSHAP with respect
                        to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> net = ImageClassifier()
                >>> neuron_grad_shap = NeuronGradientShap(net, net.linear2)
                >>> input = torch.randn(3, 3, 32, 32, requires_grad=True)
                >>> # choosing baselines randomly
                >>> baselines = torch.randn(20, 3, 32, 32)
                >>> # Computes gradient SHAP of first neuron in linear2 layer
                >>> # with respect to the input's of the network.
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = neuron_grad_shap.attribute(input, neuron_ind=0
                                                             baselines)

        """
        gs = GradientShap(self.forward_func)
        gs.gradient_func = construct_neuron_grad_fn(
            self.layer,
            neuron_index,
            self.device_ids,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )

        return gs.attribute(
            inputs,
            baselines,
            n_samples=n_samples,
            stdevs=stdevs,
            additional_forward_args=additional_forward_args,
        )
