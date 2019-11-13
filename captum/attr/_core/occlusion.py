#!/usr/bin/env python3

import torch
import numpy as np

from .._utils.common import (
    _format_attributions,
    _format_input,
    _run_forward,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
)
from .._utils.attribution import PerturbationAttribution

from .feature_ablation import FeatureAblation

class Occlusion(FeatureAblation):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        super().__init__(forward_func)
        self.use_weights = True

    def attribute(
        self,
        inputs,
        occlusion_shapes,
        baselines=None,
        target=None,
        additional_forward_args=None,
        ablations_per_eval=1,
    ):
        r""""
        A perturbation based approach to computing attribution, involving
        replacing each input feature with a given baseline / reference, and
        computing the difference in output. By default, each scalar value within
        each input tensor is taken as a feature and replaced independently. Passing
        a feature mask, allows grouping features to be ablated together. This can
        be used in cases such as images, where an entire segment or region
        can be ablated, measuring the importance of the segment (feature group).
        Each input scalar in the group will be given the same attribution value
        equal to the change in target as a result of ablating the entire feature
        group.

        Args:

                inputs (tensor or tuple of tensors):  Input for which ablation
                            attributions are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            multiple input tensors are provided, the examples must
                            be aligned appropriately.
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
                target (int, tuple, tensor or list, optional):  Output indices for
                            which difference is computed (for classification cases,
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
                            For a tensor, the first dimension of the tensor must
                            correspond to the number of examples. For all other types,
                            the given argument is used for all forward evaluations.
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
                            The attributions with respect to each input feature.
                            Attributions will always be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining FeatureAblation interpreter
            >>> ablator = FeatureAblation(net)
            >>> # Computes ablation attribution, ablating each of each of the 16
            >>> # scalar input independently.
            >>> attr = ablator.attribute(input, target=1)

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
            >>> # simultaneously, and the attribution for all inputs in the same
            >>> # group (0, 1, 2, and 3) are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = ablator.attribute(input, target=1, feature_mask=feature_mask)
        """
        formatted_inputs = _format_input(inputs)

        # Construct occlusion blocks
        if not isinstance(occlusion_shapes[0], tuple):
            occlusion_shapes = (occlusion_shapes,)
        assert len(occlusion_shapes) == len(formatted_inputs), "Must provide occlusion dimensions for each tensor."
        occlusion_tensors = tuple(torch.ones(occ_shape) for occ_shape in occlusion_shapes)

        # Construct feature masks
        feature_masks = []
        for i, inp in enumerate(formatted_inputs):
            current_shape = np.subtract(np.add(tuple(inp.shape[1:]), 1), occlusion_shapes[i])
            feature_masks.append(torch.reshape(
                    torch.arange(int(np.prod(current_shape)), device=formatted_inputs[i].device),
                    tuple(current_shape),
                ))

        # Use ablation attribute method
        return super().attribute(inputs, baselines=baselines, target=target, feature_mask=tuple(feature_masks), additional_forward_args=additional_forward_args, ablations_per_eval=ablations_per_eval, occlusion_tensors=occlusion_tensors)

    def _construct_ablated_input(
        self, feature_tensor, input_mask, baseline, start_feature, end_feature, **kwargs
    ):
        r"""
        Ablates given feature tensor with given input feature mask, feature range,
        and baselines. feature_tensor shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.
        input_mask has same number of dimensions as original input tensor (one less
        than feature_tensor), and can have first dimension either 1, applying same
        feature mask to all examples, or num_examples. baseline is expected to
        be broadcastable to match feature_tensor.

        This method returns the ablated feature tensor, which has the same
        dimensionality as feature_tensor as well as the corresponding mask with
        either the same dimensionality as feature_tensor or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        current_mask = torch.stack(
            [self._occlusion_mask(feature_tensor, input_mask, j, kwargs["occlusion_tensors"]) for j in range(start_feature, end_feature)], dim=0
        ).long()
        ablated_tensor = (feature_tensor * (1 - current_mask).float()) + (
            baseline * current_mask.float()
        )
        return ablated_tensor, current_mask

    def _occlusion_mask(self, input, mask, feature_number, occlusion_tensor):
        index = (mask == feature_number).nonzero()[0].tolist()
        val_list = []
        for i in range(len(index)-1, -1, -1):
            val_list.append(index[i])
            val_list.append(input.shape[i + 2] - index[i] - occlusion_tensor.shape[i])
        return torch.nn.functional.pad(occlusion_tensor, tuple(val_list))
