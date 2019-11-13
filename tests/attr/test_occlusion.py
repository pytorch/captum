#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.occlusion import Occlusion

from .helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_input(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._occlusion_test_assert(
            net, inp, [80.0, 200.0, 120.0], ablations_per_eval=(1, 2, 3), occlusion_shapes=((1,)),
        )

    """def test_simple_input_conv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        self._occlusion_test_assert(
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
            occlusion_shapes=((1,2,2), (1,2,2)),
            ablations_per_eval=(1,),
        )

    def test_simple_multi_input_conv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        self._occlusion_test_assert(
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
            occlusion_shapes=((1,2,2), (1,2,2)),
            ablations_per_eval=(1,),
        )"""

    def _occlusion_test_assert(
        self,
        model,
        test_input,
        expected_ablation,
        occlusion_shapes=None,
        additional_input=None,
        ablations_per_eval=(1,),
        baselines=None,
    ):
        for batch_size in ablations_per_eval:
            ablation = Occlusion(model)
            attributions = ablation.attribute(
                test_input,
                occlusion_shapes=occlusion_shapes,
                target=0,
                additional_forward_args=additional_input,
                baselines=baselines,
                ablations_per_eval=batch_size,
            )
            print("FINAL ANSWER")
            print(attributions)
            if isinstance(expected_ablation, tuple):
                for i in range(len(expected_ablation)):
                    assertTensorAlmostEqual(self, attributions[i], expected_ablation[i])
            else:
                assertTensorAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
