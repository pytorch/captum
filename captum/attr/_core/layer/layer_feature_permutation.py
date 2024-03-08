#!/usr/bin/env python3
from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation


class LayerFeaturePermutation(LayerFeatureAblation):
    r"""
    A perturbation based approach to computing layer attribution similar to
    LayerFeatureAblation, but using FeaturePermutation under the hood instead
    of FeatureAblation.
    """

    @property
    def attributor(self):
        return FeaturePermutation
