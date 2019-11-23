#!/usr/bin/env python3

import torch
import numpy as np

from .._utils.common import _format_input

from .feature_ablation import FeatureAblation


class Occlusion(FeatureAblation):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        FeatureAblation.__init__(self, forward_func)
        self.use_weights = True

    def attribute(
        self,
        inputs,
        occlusion_shapes,
        strides=None,
        baselines=None,
        target=None,
        additional_forward_args=None,
        ablations_per_eval=1,
    ):
        r""""
        A perturbation based approach to computing attribution, involving
        replacing each contiguous rectangular region with a given baseline /
        reference, and computing the difference in output. For features located
        in multiple regions (hyperrectangles), the corresponding output differences
        are averaged to compute the attribution for that feature.

        The first patch is applied with the corner aligned with all indices 0,
        and strides are applied until the entire dimension range is covered. Note
        that this may cause the final patch applied in a direction to be cut-off
        and thus smaller than the target occlusion shape.

        Args:

                inputs (tensor or tuple of tensors):  Input for which occlusion
                            attributions are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            multiple input tensors are provided, the examples must
                            be aligned appropriately.
                occlusion_shapes (tuple or tuple of tuples): Shape of patch
                            (hyperrectangle) to occlude each input. For a single
                            input tensor, this must be a tuple of length equal to the
                            number of dimensions of the input tensor - 1, defining
                            the dimensions of the patch. For multiple input tensors,
                            this must be a tuple containing one tuple for each input
                            tensor defining the dimensions of the patch for that
                            input tensor, as described for the single tensor case.
                strides (int or tuple or tuple of ints or tuple of tuples, optional):
                            This defines the step by which the occlusion hyperrectangle
                            should be shifted by in each direction for each iteration.
                            For a single tensor input, this can be either a single
                            integer, which is used as the step size in each direction,
                            or a tuple of integers matching the number of dimensions
                            in the occlusion shape, defining the step size in the
                            corresponding dimension. For multiple tensor inputs, this
                            can be either a tuple of integers, one for each input
                            tensor (used for all dimensions of the corresponding
                            tensor), or a tuple of tuples, providing the stride per
                            dimension for each tensor.
                            If None is provided, a stride of 1 is used for each
                            dimension of each input tensor.
                            Default: None
                baselines (scalar, tensor, tuple of scalars or tensors, optional):
                            Baselines define reference value which replaces each
                            feature when occluded.
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
                ablations_per_eval (int, optional): Allows multiple occlusions
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
            >>> ablator = Occlusion(net)
            >>> # Computes occlusion attribution, ablating each 1x3x3 patch,
            >>> # shifting in each direction by the default of 1.
            >>> attr = ablator.attribute(input, target=1, occlusion_shapes=(1,3,3))
        """
        formatted_inputs = _format_input(inputs)

        # Formatting strides
        if strides is None:
            strides = tuple(1 for input in formatted_inputs)
        if len(formatted_inputs) == 1 and not (
            isinstance(strides, tuple) and len(strides) == 1
        ):
            strides = (strides,)
        assert isinstance(strides, tuple) and len(strides) == len(
            formatted_inputs
        ), "Strides must be provided for each input tensor."

        # Construct occlusion blocks
        if not isinstance(occlusion_shapes[0], tuple):
            occlusion_shapes = (occlusion_shapes,)
        assert len(occlusion_shapes) == len(
            formatted_inputs
        ), "Must provide occlusion dimensions for each tensor."
        occlusion_tensors = tuple(
            torch.ones(occ_shape, device=formatted_inputs[i].device)
            for i, occ_shape in enumerate(occlusion_shapes)
        )

        # Construct counts, defining number of steps to make of occlusion block in
        # each dimension.
        shift_counts = []
        for i, inp in enumerate(formatted_inputs):
            current_shape = np.subtract(inp.shape[1:], occlusion_shapes[i])
            shift_counts.append(
                tuple(
                    np.add(np.ceil(np.divide(current_shape, strides[i])).astype(int), 1)
                )
            )

        # Use ablation attribute method
        return super().attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            ablations_per_eval=ablations_per_eval,
            occlusion_tensors=occlusion_tensors,
            shift_counts=tuple(shift_counts),
            strides=strides,
        )

    def _construct_ablated_input(
        self, feature_tensor, input_mask, baseline, start_feature, end_feature, **kwargs
    ):
        r"""
        Ablates given feature tensor with given input feature mask, feature range,
        and baselines, and any additional arguments.
        feature_tensor shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.

        input_mask is None for occlusion, and the mask is constructed
        using occlusion_tensors, strides, and shift counts, provided in
        kwargs. baseline is expected to
        be broadcastable to match feature_tensor.

        This method returns the ablated feature tensor, which has the same
        dimensionality as feature_tensor as well as the corresponding mask with
        either the same dimensionality as feature_tensor or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        current_mask = torch.stack(
            [
                self._occlusion_mask(
                    feature_tensor,
                    j,
                    kwargs["occlusion_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()
        ablated_tensor = (feature_tensor * (1 - current_mask).float()) + (
            baseline * current_mask.float()
        )
        return ablated_tensor, current_mask

    def _occlusion_mask(
        self, input, feature_number, occlusion_tensor, strides, shift_counts
    ):
        current_total = feature_number
        current_index = []
        for i, val in enumerate(shift_counts):
            stride = strides[i] if isinstance(strides, tuple) else strides
            current_index.append((current_total % val) * stride)
            current_total = current_total // val

        remaining_padding = np.subtract(
            input.shape[2:], np.add(current_index, occlusion_tensor.shape)
        )
        pad_values = [
            val for pair in zip(remaining_padding, current_index) for val in pair
        ]
        pad_values.reverse()
        padded_tensor = torch.nn.functional.pad(occlusion_tensor, tuple(pad_values))
        return padded_tensor.reshape((1,) + padded_tensor.shape)

    def _get_feature_range_and_mask(self, input, input_mask, **kwargs):
        feature_max = np.prod(kwargs["shift_counts"])
        return 0, feature_max, None
