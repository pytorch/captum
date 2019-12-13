#!/usr/bin/env python3
import torch

from .feature_ablation import FeatureAblation


def _permute_feature(x, feature_mask):
    n = x.size(0)
    assert n > 1, "cannot permute features with batch_size = 1"

    perm = torch.randperm(n)
    no_perm = torch.arange(n)
    while (perm == no_perm).all():
        perm = torch.randperm(n)

    return (x[perm] * feature_mask.to(dtype=x.dtype)) + (
        x * feature_mask.bitwise_not().to(dtype=x.dtype)
    )


class PermutationFeatureImportance(FeatureAblation):
    def __init__(self, forward_func=None, perm_fn=_permute_feature):
        FeatureAblation.__init__(self, forward_func=forward_func)
        self.perm_fn = perm_fn

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        feature_mask=None,
        ablations_per_eval=1,
    ):
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
        self, feature_tensor, input_mask, baseline, start_feature, end_feature
    ):
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
                for x, mask in zip(feature_tensor, current_mask)
            ]
        )
        return output, current_mask
