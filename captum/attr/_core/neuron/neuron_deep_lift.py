#!/usr/bin/env python3
from ..._utils.attribution import NeuronAttribution, GradientAttribution
from ..._utils.gradient import construct_neuron_grad_fn

from ..deep_lift import DeepLift, DeepLiftShap

from typing import Callable, Optional, Tuple, Union, Any
from torch import Tensor
from torch.nn import Module
from ..._utils.typing import TensorOrTupleOfTensors


class NeuronDeepLift(NeuronAttribution, GradientAttribution):
    def __init__(self, model: Module, layer: Module) -> None:
        r"""
        Args:

            model (torch.nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                          Attributions for a particular neuron for the input or output
                          of this layer are computed using the argument neuron_index
                          in the attribute method.
                          Currently, it is assumed that the inputs or the outputs
                          of the layer, depending on which one is used for
                          attribution, can only be a single tensor.
        """
        NeuronAttribution.__init__(self, model, layer)
        GradientAttribution.__init__(self, model)

    def attribute(
        self,
        inputs: TensorOrTupleOfTensors,
        neuron_index: Union[int, Tuple[int, ...]],
        baselines: Optional[
            Union[Tensor, int, float, Tuple[Union[Tensor, int, float], ...]]
        ] = None,
        additional_forward_args: Any = None,
        attribute_to_neuron_input: bool = False,
        custom_attribution_func: Optional[Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensors:
        r""""
        Implements DeepLIFT algorithm for the neuron based on the following paper:
        Learning Important Features Through Propagating Activation Differences,
        Avanti Shrikumar, et. al.
        https://arxiv.org/abs/1704.02685

        and the gradient formulation proposed in:
        Towards better understanding of gradient-based attribution methods for
        deep neural networks,  Marco Ancona, et.al.
        https://openreview.net/pdf?id=Sy21R9JAW

        This implementation supports only Rescale rule. RevealCancel rule will
        be supported in later releases.
        Although DeepLIFT's(Rescale Rule) attribution quality is comparable with
        Integrated Gradients, it runs significantly faster than Integrated
        Gradients and is preferred for large datasets.

        Currently we only support a limited number of non-linear activations
        but the plan is to expand the list in the future.

        Note: As we know, currently we cannot access the building blocks,
        of PyTorch's built-in LSTM, RNNs and GRUs such as Tanh and Sigmoid.
        Nonetheless, it is possible to build custom LSTMs, RNNS and GRUs
        with performance similar to built-in ones using TorchScript.
        More details on how to build custom RNNs can be found here:
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Args:

            inputs (tensor or tuple of tensors):  Input for which layer
                        attributions are computed. If forward_func takes a
                        single tensor as input, a single input tensor should be
                        provided. If forward_func takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            neuron_index (int or tuple): Index of neuron in output of given
                        layer for which attribution is desired. Length of
                        this tuple must be one less than the number of
                        dimensions in the output of the given layer (since
                        dimension 0 corresponds to number of examples).
                        An integer may be provided instead of a tuple of
                        length 1.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

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

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided
                        to forward_func in order, following the arguments in inputs.
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
            custom_attribution_func (callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:
                            - custom_attribution_func(multipliers)
                            - custom_attribution_func(multipliers, inputs)
                            - custom_attribution_func(multipliers, inputs, baselines)
                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Computes attributions using Deeplift's rescale rule for
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
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = NeuronDeepLift(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and neuron
            >>> # index (4,1,2).
            >>> attribution = dl.attribute(input, (4,1,2))
        """
        dl = DeepLift(self.forward_func)
        dl.gradient_func = construct_neuron_grad_fn(
            self.layer,
            neuron_index,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )

        return dl.attribute(
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            custom_attribution_func=custom_attribution_func,
        )


class NeuronDeepLiftShap(NeuronAttribution, GradientAttribution):
    def __init__(self, model: Module, layer: Module) -> None:
        r"""
        Args:

            model (torch.nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                          Attributions for a particular neuron for the input or output
                          of this layer are computed using the argument neuron_index
                          in the attribute method.
                          Currently, only layers with a single tensor input and output
                          are supported.
        """
        NeuronAttribution.__init__(self, model, layer)
        GradientAttribution.__init__(self, model)

    def attribute(
        self,
        inputs: TensorOrTupleOfTensors,
        neuron_index: Union[int, Tuple[int, ...]],
        baselines: Optional[
            Union[
                int,
                float,
                Tensor,
                Tuple[Union[Tensor, int, float], ...],
                Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
            ]
        ] = None,
        additional_forward_args: Any = None,
        attribute_to_neuron_input: bool = False,
        custom_attribution_func: Optional[Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensors:
        r"""
        Extends NeuronAttribution and uses LayerDeepLiftShap algorithms and
        approximates SHAP values for given input `layer` and `neuron_index`.
        For each input sample - baseline pair it computes DeepLift attributions
        with respect to inputs or outputs of given `layer` and `neuron_index`
        averages resulting attributions across baselines. Whether to compute the
        attributions with respect to the inputs or outputs of the layer is defined
        by the input flag `attribute_to_layer_input`.
        More details about the algorithm can be found here:

        http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

        Note that the explanation model:
            1. Assumes that input features are independent of one another
            2. Is linear, meaning that the explanations are modeled through
               the additive composition of feature effects.
        Although, it assumes a linear model for each explanation, the overall
        model across multiple explanations can be complex and non-linear.
        Args:

            inputs (tensor or tuple of tensors):  Input for which layer
                        attributions are computed. If forward_func takes a
                        single tensor as input, a single input tensor should be
                        provided. If forward_func takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            neuron_index (int or tuple): Index of neuron in output of given
                        layer for which attribution is desired. Length of
                        this tuple must be one less than the number of
                        dimensions in the output of the given layer (since
                        dimension 0 corresponds to number of examples).
                        An integer may be provided instead of a tuple of
                        length 1.
            baselines (tensor, tuple of tensors, callable):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references. Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                            the first dimension equal to the number of examples
                            in the baselines' distribution. The remaining dimensions
                            must match with input tensor's dimension starting from
                            the second dimension.

                        - a tuple of tensors, if inputs is a tuple of tensors,
                            with the first dimension of any tensor inside the tuple
                            equal to the number of examples in the baseline's
                            distribution. The remaining dimensions must match
                            the dimensions of the corresponding input tensor
                            starting from the second dimension.

                        - callable function, optionally takes `inputs` as an
                            argument and either returns a single tensor
                            or a tuple of those.

                        It is recommended that the number of samples in the baselines'
                        tensors is larger than one.
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided
                        to forward_func in order, following the arguments in inputs.
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
            custom_attribution_func (callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:
                            - custom_attribution_func(multipliers)
                            - custom_attribution_func(multipliers, inputs)
                            - custom_attribution_func(multipliers, inputs, baselines)
                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Computes attributions using Deeplift's rescale rule for
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
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = NeuronDeepLiftShap(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and neuron
            >>> # index (4,1,2).
            >>> attribution = dl.attribute(input, (4,1,2))
        """
        dl = DeepLiftShap(self.forward_func)
        dl.gradient_func = construct_neuron_grad_fn(
            self.layer,
            neuron_index,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )

        return dl.attribute(
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            custom_attribution_func=custom_attribution_func,
        )
