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
from .helpers.utils import assertTensorTuplesAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_shapley_sampling(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._shapley_test_assert(
            net,
            inp,
            [76.66666, 196.66666, 116.66666],
            ablations_per_eval=(1, 2, 3),
            n_samples=250,
        )

    def test_simple_shapley_sampling_with_mask(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._shapley_test_assert(
            net,
            inp,
            [275.0, 275.0, 115.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            ablations_per_eval=(1, 2, 3),
        )

    def test_simple_shapley_sampling_with_baselines(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]])
        self._shapley_test_assert(
            net,
            inp,
            [280.0, 280.0, 120.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            ablations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]])
        self._shapley_test_assert(
            net,
            inp,
            [[7.0, 32.5, 10.5], [76.66666, 196.66666, 116.66666]],
            ablations_per_eval=(1, 2, 3),
            delta=0.5,
            n_samples=200,
        )

    def test_multi_sample_ablation_with_mask(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._shapley_test_assert(
            net,
            inp,
            [[39.5, 39.5, 10.5], [275.0, 275.0, 115.0]],
            feature_mask=mask,
            ablations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation_with_mask(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected = (
            [[1088.6666, 1088.6666, 1088.6666], [255.0, 595.0, 255.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 595.0, 0.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 255.0, 255.0]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            delta=0.8,
        )
        expected_with_baseline = (
            [[1092, 1092, 1092], [260, 600.0, 260]],
            [[80, 1092, 160], [260, 600.0, 0]],
            [[80, 1092, 160], [260, 260, 260]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            ablations_per_eval=(1, 2, 3),
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
        n_samples=100,
        delta=0.5,
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
                n_samples=n_samples,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_ablation, delta=delta, mode="max"
            )


if __name__ == "__main__":
    unittest.main()
