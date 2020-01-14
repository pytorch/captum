#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.shapley_sampling import ShapleyValueSampling

from .helpers.basic_models import (
    BasicModel,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_shapley_sampling(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._shapley_test_assert(
            net, inp, [80.0, 200.0, 120.0], ablations_per_eval=(1, 2, 3)
        )

    def _shapley_test_assert(
        self,
        model,
        test_input,
        expected_ablation,
        feature_mask=None,
        additional_input=None,
        ablations_per_eval=(1,),
        baselines=None,
        target=0,
    ):
        for batch_size in ablations_per_eval:
            shapley_samp = ShapleyValueSampling(model)
            attributions = shapley_samp.attribute(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                ablations_per_eval=batch_size,
            )
            if isinstance(expected_ablation, tuple):
                for i in range(len(expected_ablation)):
                    assertTensorAlmostEqual(self, attributions[i], expected_ablation[i])
            else:
                assertTensorAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
