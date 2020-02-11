#!/usr/bin/env python3
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .feature_ablation import FeatureAblation
from .._utils.typing import TensorOrTupleOfTensors


def _permute_feature(x: Tensor, feature_mask: Tensor) -> Tensor:
    n = x.size(0)
    assert n > 1, "cannot permute features with batch_size = 1"

    perm = torch.randperm(n)
    no_perm = torch.arange(n)
    while (perm == no_perm).all():
        perm = torch.randperm(n)

    return (x[perm] * feature_mask.to(dtype=x.dtype)) + (
        x * feature_mask.bitwise_not().to(dtype=x.dtype)
    )


class FeaturePermutation(FeatureAblation):
    def __init__(self, forward_func: Callable, perm_func: Callable = _permute_feature):
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
            perm_func (callable): A function that accepts a batch of inputs and a feature mask, 
                                  and "permutes" the feature across the batch. NOTE: one obviously does 
                                  not have to perform a permutation. See `_permute_feature` as an example on how 
                                  to implement your own permutation function.
                                  Default: `_permute_feature`
        """
        FeatureAblation.__init__(self, forward_func=forward_func)
        self.perm_func = perm_func

    # suppressing error caused by the child class not having a matching
    # signature to the parent
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensors,
        target: Optional[
            Union[int, Tuple[int, ...], Tensor, List[Tuple[int, ...]]]
        ] = None,
        additional_forward_args: Any = None,
        feature_mask: Optional[TensorOrTupleOfTensors] = None,
        ablations_per_eval: int = 1,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensors:
        r"""
        This attribution method essentially implements the permutation feature importance algorithm, 
        as described here: https://christophm.github.io/interpretable-ml-book/feature-importance.html

        A basic tl;dr of the algorithm is:

        perm_feature_importance(batch):
            importance = dict()
            baseline_error = error_metric(model(batch), batch_labels)
            for each feature:
                permute this feature across the batch
                error = error_metric(model(permuted_batch), batch_labels)
                importance[feature] = baseline_error - error
                "un-permute" the feature across the batch

            sort importance in descending order
            return importance

        It should be noted that the `error_metric` must be called in the `forward_func`. You do
        not need to provide an error metric, e.g. you could simply return the logits (the model output), but this may or may not provide a meaningful attribution.

        This function is almost equivalent to `FeatureAblation.attribute`. The main difference
        is the way ablated examples are generated. Specifically they are generated through the 
        `perm_func`, as we set the baselines for `FeatureAblation.attribute` to None.


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
                additional_forward_args (any, optional): If the forward function
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

                            The first dimension of each mask must be 1, as we require
                            to have the same group of features for each input sample.
                            TODO

                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature, which
                            is permuted independently.
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
                            If the forward function returns a single scalar per batch,
                            ablations_per_eval must be set to 1.
                            Default: 1
                **kwargs (Any, optional): Any additional arguments used by child
                            classes of FeatureAblation (such as Occlusion) to construct
                            ablations. These arguments are ignored when using
                            FeatureAblation directly.
                            Default: None
        """
        return FeatureAblation.attribute(
            self,
            inputs,
            baselines=None,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            ablations_per_eval=ablations_per_eval,
            **kwargs,
        )

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Tensor,
        baseline: Union[int, float, Tensor],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        This function ablates `expanded_input` with a given feature mask and feature range. Ablation
        occurs via calling `self.perm_func` across each batch within `expanded_input`. As 
        with `FeatureAblation._construct_ablated_input`:
        - `expanded_input.shape = (num_features, num_examples, ...)`
        - `num_features = end_feature - start_feature` (i.e. start and end is a half-closed interval)
        - `input_mask` is a tensor of the same shape as one input, which describes the locations of each feature via their "index"
        
        Since `baselines` is set to None for `FeatureAblation.attribute, this will be the zero tensor, however, it is not used.
        """
        assert input_mask.shape[0] == 1, (
            "input_mask.shape[0] != 1: pass in one mask in order to permute"
            "the same features for each input"
        )
        current_mask = torch.stack(
            [input_mask == j for j in range(start_feature, end_feature)], dim=0
        ).bool()

        output = torch.stack(
            [
                self.perm_func(x, mask.squeeze(0))
                for x, mask in zip(expanded_input, current_mask)
            ]
        )
        return output, current_mask
