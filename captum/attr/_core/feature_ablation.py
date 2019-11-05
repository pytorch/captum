#!/usr/bin/env python3

import torch

from .._utils.common import (
    _format_attributions,
    format_input,
    _run_forward,
    _expand_additional_forward_args,
    _format_additional_forward_args,
)
from .._utils.attribution import PerturbationAttribution


class FeatureAblation(PerturbationAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        feature_mask=None,
        baselines=None,
        internal_batch_size=None,
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
                            should contain the same number of tensors as inputs,
                            and features within each input tensor are ablated
                            independetly (not across tensors). Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Each tensor
                            should contain integers in the range 0 to num_features
                            - 1, and indices corresponding to the same feature should
                            have the same value.
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature, which
                            is ablated independently.
                            Default: None
                baselines (scalar, tensor, tuple of scalars or tensors, optional):
                            Baselines define reference value which replaces each feature
                            when ablated. In order to assign attribution scores DeepLift
                            computes the differences between the inputs and references
                            and corresponding outputs.
                            Baselines can be provided either as :

                            - a single tensor, if inputs is a single tensor, with
                                exactly the same dimensions as inputs (first dimension
                                can be 1, which applies the same baseline to all
                                input examples).

                            - a tuple of tensors, if inputs is a tuple of tensors,
                                with matching dimensions to inputs (first dimension
                                can be 1, which applies the same baseline to all
                                input examples).

                            - a single scalar, if inputs is a single tensor, which will
                                be broadcasted for each input value in input tensor.

                            - a tuple of scalars, if inputs is a tuple of tensors, with
                                exactly the same number of elements as inputs tuple.
                                Each scalar element in baselines' tuple is broadcasted
                                for each input tensor at the same index in inputs
                                tuple.
                            Default: zero scalar for each input tensor
                internal_batch_size (int, optional): Divides total #features *
                            #examples data points into chunks of size
                            internal_batch_size, which are computed sequentially.
                            internal_batch_size should be a multiple of #examples.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain internal_batch_size / num_devices examples.
                            If internal_batch_size is None, then all evaluations are
                            processed in one batch.
                            Default: None

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

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # Generating random input with size 2x3x3x32
            >>> input = torch.randn(2, 3, 32, 32)
            >>> # Defining FeatureAblation interpreter
            >>> ablator = FeatureAblation(net)
            >>> # Computes ablation attribution, ablating each scalar input
            >>> # independently.
            >>> attribution = ablator.attribute(input, target=3)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        num_examples = inputs[0].shape[0]
        feature_mask = format_input(feature_mask) if feature_mask is not None else None
        assert (
            internal_batch_size is None or internal_batch_size >= inputs[0].shape[0]
        ), "Internal batch size must be at least number of examples given."

        # Computes initial evaluation with all features, which is compared
        # to each ablated result.
        initial_eval = _run_forward(
            self.forward_func, inputs, target, additional_forward_args
        )
        assert (
            initial_eval[0].numel() == 1
        ), "Target should identify a single element in the model output."
        initial_eval = initial_eval.reshape((1, num_examples))

        # Initialize attribution totals and counts
        total_attrib = [torch.zeros_like(input) for input in inputs]
        weights = [torch.zeros_like(input) for input in inputs]

        # Iterate through each feature tensor for ablation
        for i in range(len(inputs)):
            # Obtain feature mask for selected input tensor
            input_mask = (
                torch.reshape(
                    torch.arange(torch.numel(inputs[i][0])), inputs[i][0:1].shape
                )
                if feature_mask is None
                else feature_mask[i]
            )
            for (
                current_inputs,
                current_add_args,
                current_mask,
            ) in self._ablation_generator(
                i,
                inputs,
                additional_forward_args,
                baselines,
                input_mask,
                internal_batch_size,
            ):
                modified_eval = _run_forward(
                    self.forward_func, current_inputs, target, current_add_args
                )
                eval_diff = (
                    initial_eval - modified_eval.reshape((-1, num_examples))
                ).reshape((-1, num_examples) + (len(inputs[i].shape) - 1) * (1,))
                weights[i] += current_mask.float().sum(dim=0)
                total_attrib[i] += (eval_diff * current_mask.float()).sum(dim=0)

        # Divide total attributions by counts and return formatted attributions
        div_attrib = tuple(
            single_attrib / weight
            for single_attrib, weight in zip(total_attrib, weights)
        )
        return _format_attributions(is_inputs_tuple, div_attrib)

    def _ablation_generator(
        self, i, inputs, additional_args, baselines, input_mask, internal_batch_size
    ):
        num_features = torch.max(input_mask).item() + 1
        num_examples = inputs[0].shape[0]
        batch_size = (
            num_features
            if internal_batch_size is None
            else min(internal_batch_size // num_examples, num_features)
        )
        baseline = baselines[i] if isinstance(baselines, tuple) else baselines
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.reshape((1,) + baseline.shape)

        # Repeat features and additional args for batch size.
        all_features_repeated = [
            torch.cat([inputs[j]] * batch_size, dim=0) for j in range(len(inputs))
        ]
        additional_args_repeated = (
            _expand_additional_forward_args(additional_args, batch_size)
            if additional_args is not None
            else None
        )

        num_features_processed = torch.min(input_mask).item()
        while num_features_processed < num_features:
            current_batch = min(batch_size, num_features - num_features_processed)

            # Store appropriate inputs and additional args based on batch size.
            if current_batch != batch_size:
                current_features = [
                    tensor[0 : current_batch * num_examples]
                    for tensor in all_features_repeated
                ]
                current_additional_args = (
                    _expand_additional_forward_args(additional_args, current_batch)
                    if additional_args is not None
                    else None
                )
            else:
                current_features = all_features_repeated
                current_additional_args = additional_args_repeated

            # Store existing tensor before modifying
            original_tensor = current_features[i]
            # Construct ablated batch for features in range num_features_processed
            # to num_features_processed + current_batch and return mask with
            # same size as ablated batch.
            ablated_features, current_mask = self._construct_ablated_input(
                current_features[i].reshape(
                    (current_batch, num_examples) + current_features[i].shape[1:]
                ),
                input_mask,
                baseline,
                num_features_processed,
                num_features_processed + current_batch,
            )
            current_features[i] = ablated_features.reshape(
                (-1,) + ablated_features.shape[2:]
            )
            yield tuple(current_features), current_additional_args, current_mask
            # Replace existing tensor at index i.
            current_features[i] = original_tensor
            num_features_processed += current_batch

    def _construct_ablated_input(
        self, feature_tensor, input_mask, baseline, start_feature, end_feature
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
        thus counted towards ablations for that feature).
        """
        current_mask = torch.stack(
            [input_mask == j for j in range(start_feature, end_feature)], dim=0
        ).long()
        ablated_tensor = (feature_tensor * (1 - current_mask).float()) + (
            baseline * current_mask.float()
        )
        return ablated_tensor, current_mask
