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
    def __init__(
        self, forward_func: Callable, perm_fn: Callable = _permute_feature,
    ):
        FeatureAblation.__init__(self, forward_func=forward_func)
        self.perm_fn = perm_fn

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
        **kwargs: Any
    ) -> TensorOrTupleOfTensors:
        return FeatureAblation.attribute(
            self,
            inputs,
            baselines=None,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            ablations_per_eval=ablations_per_eval,
        )

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Tensor,
        baseline: Union[int, float, Tensor],
        start_feature: int,
        end_feature: int,
        **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        assert input_mask.shape[0] == 1, (
            "input_mask.shape[0] != 1: pass in one mask in order to permute"
            "the same features for each input"
        )
        current_mask = torch.stack(
            [input_mask == j for j in range(start_feature, end_feature)], dim=0
        ).bool()

        output = torch.stack(
            [
                self.perm_fn(x, mask.squeeze(0))
                for x, mask in zip(expanded_input, current_mask)
            ]
        )
        return output, current_mask
