#!/usr/bin/env python3
import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable, List, Optional, Tuple, Union, Any


from ..._utils.attribution import NeuronAttribution, PerturbationAttribution
from ..._utils.common import _verify_select_column
from ..._utils.gradient import _forward_layer_eval
from ..._utils.typing import TensorOrTupleOfTensors

from ..feature_ablation import FeatureAblation


class NeuronFeatureAblation(NeuronAttribution, PerturbationAttribution):
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
                          Attributions for a particular neuron in the input or output
                          of this layer are computed using the argument neuron_index
                          in the attribute method.
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
        PerturbationAttribution.__init__(self, forward_func)

    def attribute(
        self,
        inputs: TensorOrTupleOfTensors,
        neuron_index: Union[int, Tuple[int, ...]],
        baselines: Optional[
            Union[Tensor, int, float, Tuple[Union[Tensor, int, float], ...]]
        ] = None,
        additional_forward_args: Any = None,
        feature_mask: Optional[TensorOrTupleOfTensors] = None,
        attribute_to_neuron_input: bool = False,
        ablations_per_eval: int = 1,
    ) -> TensorOrTupleOfTensors:
        r"""
            A perturbation based approach to computing neuron attribution,
            involving replacing each input feature with a given baseline /
            reference, and computing the difference in the neuron's input / output.
            By default, each scalar value within
            each input tensor is taken as a feature and replaced independently. Passing
            a feature mask, allows grouping features to be ablated together. This can
            be used in cases such as images, where an entire segment or region
            can be ablated, measuring the importance of the segment (feature group).
            Each input scalar in the group will be given the same attribution value
            equal to the change in target as a result of ablating the entire feature
            group.


            Args:

                inputs (tensor or tuple of tensors):  Input for which neuron
                            attributions are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if multiple input tensors
                            are provided, the examples must be aligned appropriately.
                neuron_index (int or tuple): Index of neuron in output of given
                              layer for which attribution is desired. The length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).
                              An integer may be provided instead of a tuple of
                              length 1.
                baselines (scalar, tensor, tuple of scalars or tensors, optional):
                            Baselines define reference value which replaces each
                            feature when ablated.
                            Baselines can be provided as:
                            - a single tensor, if inputs is a single tensor, with
                                exactly the same dimensions as inputs or
                                broadcastable to match the dimensions of inputs
                            - a single scalar, if inputs is a single tensor, which will
                                be broadcasted for each input value in input tensor.
                            - a tuple of tensors or scalars, the baseline corresponding
                                to each tensor in the inputs' tuple can be:
                                - either a tensor with
                                    exactly the same dimensions as inputs or
                                    broadcastable to match the dimensions of inputs
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
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                feature_mask (tensor or tuple of tensors, optional):
                            feature_mask defines a mask for the input, grouping
                            features which should be ablated together. feature_mask
                            should contain the same number of tensors as inputs.
                            Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Each tensor
                            should contain integers in the range 0 to num_features
                            - 1, and indices corresponding to the same feature should
                            have the same value.
                            Note that features within each input tensor are ablated
                            independently (not across tensors).
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature, which
                            is ablated independently.
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
                ablations_per_eval (int, optional): Allows ablation of multiple features
                            to be processed simultaneously in one call to forward_fn.
                            Each forward pass will contain a maximum of
                            ablations_per_eval * #examples samples.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain at most
                            (ablations_per_eval * #examples) / num_devices
                            samples.
                            Default: 1

            Returns:
                *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                            Attributions of particular neuron with respect to each input
                            feature. Attributions will always be the same size as the
                            provided inputs, with each value providing the attribution
                            of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.

        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x3x3.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining NeuronFeatureAblation interpreter
            >>> ablator = NeuronFeatureAblation(net, net.conv1)
            >>> # To compute neuron attribution, we need to provide the neuron
            >>> # index for which attribution is desired. Since the layer output
            >>> # is Nx12x3x3, we need a tuple in the form (0..11,0..2,0..2)
            >>> # which indexes a particular neuron in the layer output.
            >>> # For this example, we choose the index (4,1,2).
            >>> # Computes neuron gradient for neuron with
            >>> # index (4,1,2).
            >>> # Computes ablation attribution, ablating each of the 16
            >>> # scalar inputs independently.
            >>> attr = ablator.attribute(input, neuron_index=(4,1,2))

            >>> # Alternatively, we may want to ablate features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and ablating them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are ablated
            >>> # simultaneously, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = ablator.attribute(input, neuron_index=(4,1,2),
            >>>                          feature_mask=feature_mask)
        """

        def neuron_forward_func(*args: Any):
            with torch.no_grad():
                layer_eval, _ = _forward_layer_eval(
                    self.forward_func,
                    args,
                    self.layer,
                    device_ids=self.device_ids,
                    attribute_to_layer_input=attribute_to_neuron_input,
                )
                assert len(layer_eval) == 1, (
                    "Layers with multiple inputs /"
                    " outputs are not supported for neuron ablation."
                )
                return _verify_select_column(layer_eval[0], neuron_index)

        ablator = FeatureAblation(neuron_forward_func)

        return ablator.attribute(
            inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            ablations_per_eval=ablations_per_eval,
        )
