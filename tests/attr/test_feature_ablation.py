#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.feature_ablation import FeatureAblation

from .helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_ablation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net, inp, [80.0, 200.0, 120.0], ablations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_with_mask(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [280.0, 280.0, 120.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            ablations_per_eval=(1, 2, 3),
        )

    def test_simple_ablation_with_baselines(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [248.0, 248.0, 104.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            ablations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [[8.0, 35.0, 12.0], [80.0, 200.0, 120.0]],
            ablations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation_with_mask(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._ablation_test_assert(
            net,
            inp,
            [[41.0, 41.0, 12.0], [280.0, 280.0, 120.0]],
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
            [[492.0, 492.0, 492.0], [200.0, 200.0, 200.0]],
            [[80.0, 200.0, 120.0], [0.0, 400.0, 0.0]],
            [[0.0, 400.0, 40.0], [60.0, 60.0, 60.0]],
        )
        self._ablation_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        self._ablation_test_assert(
            net,
            (inp1, inp2),
            expected[0:1],
            additional_input=(inp3, 1),
            feature_mask=(mask1, mask2, mask3),
            ablations_per_eval=(1, 2, 3),
        )
        expected_with_baseline = (
            [[468.0, 468.0, 468.0], [184.0, 192.0, 184.0]],
            [[68.0, 188.0, 108.0], [-12.0, 388.0, -12.0]],
            [[-16.0, 384.0, 24.0], [12.0, 12.0, 12.0]],
        )
        self._ablation_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            ablations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        baseline1 = torch.tensor([[3.0, 0.0, 0.0]])
        baseline2 = torch.tensor([[0.0, 1.0, 0.0]])
        baseline3 = torch.tensor([[1.0, 2.0, 3.0]])
        self._ablation_test_assert(
            net,
            (inp1, inp2, inp3),
            (
                [[80.0, 400.0, 0.0], [68.0, 200.0, 120.0]],
                [[80.0, 196.0, 120.0], [0.0, 396.0, 0.0]],
                [[-4.0, 392.0, 28.0], [4.0, 32.0, 0.0]],
            ),
            additional_input=(1,),
            baselines=(baseline1, baseline2, baseline3),
            ablations_per_eval=(1, 2, 3),
        )
        baseline1_exp = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 2.0]])
        baseline2_exp = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 4.0]])
        baseline3_exp = torch.tensor([[3.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
        self._ablation_test_assert(
            net,
            (inp1, inp2, inp3),
            (
                [[80.0, 400.0, 0.0], [68.0, 200.0, 112.0]],
                [[80.0, 196.0, 120.0], [0.0, 396.0, -16.0]],
                [[-12.0, 392.0, 24.0], [4.0, 32.0, 0.0]],
            ),
            additional_input=(1,),
            baselines=(baseline1_exp, baseline2_exp, baseline3_exp),
            ablations_per_eval=(1, 2, 3),
        )

    def test_simple_multi_input_conv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        self._ablation_test_assert(
            net,
            (inp, inp2),
            (67 * torch.ones_like(inp), 13 * torch.ones_like(inp2)),
            feature_mask=(torch.tensor(0), torch.tensor(1)),
            ablations_per_eval=(1, 2, 4, 8, 12, 16),
        )
        self._ablation_test_assert(
            net,
            (inp, inp2),
            (
                [
                    [0.0, 2.0, 4.0, 3.0],
                    [4.0, 9.0, 10.0, 7.0],
                    [4.0, 13.0, 14.0, 11.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ),
            ablations_per_eval=(1, 3, 7, 14),
        )

    def _ablation_test_assert(
        self,
        model,
        test_input,
        expected_ablation,
        feature_mask=None,
        additional_input=None,
        ablations_per_eval=(1,),
        baselines=None,
    ):
        for batch_size in ablations_per_eval:
            ablation = FeatureAblation(model)
            attributions = ablation.attribute(
                test_input,
                target=0,
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
