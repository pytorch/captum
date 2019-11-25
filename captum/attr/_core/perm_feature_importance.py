#!/usr/bin/env python3
import torch

from .feature_ablation import FeatureAblation


def permute_feature(x, feature_mask):
    n = x.size(0)
    perm = torch.randperm(n)
    no_perm = torch.arange(n)
    while (perm == no_perm).all():
        perm = torch.randperm(n)

    out = x.clone()
    feature_mask = feature_mask.squeeze(0)
    for i, j in enumerate(perm):
        out[i, feature_mask] = x[j, feature_mask]

    return out


class PermutationFeatureImportance(FeatureAblation):
    def __init__(self, forward_func=None, perm_fn=permute_feature):
        super().__init__(forward_func=forward_func)
        self.perm_fn = perm_fn

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        feature_mask=None,
        ablations_per_eval=1,
        abs_attributions=False
    ):
        attribs = super().attribute(
            inputs,
            baselines=None,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            ablations_per_eval=ablations_per_eval,
        )

        f = torch.abs if abs_attributions else lambda x: x

        if isinstance(attribs, tuple):
            return tuple([f(a) for a in attribs])

        return f(attribs)

    def _construct_ablated_input(
        self, feature_tensor, input_mask, baseline, start_feature, end_feature
    ):
        current_mask = torch.stack(
            [input_mask == j for j in range(start_feature, end_feature)], dim=0
        ).bool()

        output = []
        for x, feature_mask in zip(feature_tensor, current_mask):
            # TODO: support multiple permutations
            output.append(self.perm_fn(x, feature_mask))

        output = torch.stack(output)

        return output, current_mask
