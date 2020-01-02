#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation

from ..helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from ..helpers.utils import assertTensorTuplesAlmostEqual, BaseTest


class Test(BaseTest):
    def test_simple_ablation_with_mask(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            net.linear0,
            inp,
            ([280.0, 280.0, 120.0],),
            layer_mask=torch.tensor([[0, 0, 1]]),
            ablations_per_eval=(1, 2, 3),
            attribute_to_layer_input=True,
        )

    def test_multi_input_ablation(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        baseline = torch.tensor([[1.0, 2.0, 3.0]])
        self._ablation_test_assert(
            net,
            net.model.linear1,
            (inp1, inp2, inp3),
            ([[168.0, 992.0, 148.0], [84.0, 632.0, 120.0]],),
            additional_input=(1,),
            baselines=baseline,
            ablations_per_eval=(1, 2, 3),
            attribute_to_layer_input=True,
        )
        self._ablation_test_assert(
            net,
            net.model.linear0,
            (inp1, inp2, inp3),
            [[168.0, 992.0, 148.0], [84.0, 632.0, 120.0]],
            additional_input=(1,),
            baselines=baseline,
            ablations_per_eval=(1, 2, 3),
            attribute_to_layer_input=False,
        )

    def test_multi_input_ablation_with_layer_mask(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        baseline = torch.tensor([[1.0, 2.0, 3.0]])
        layer_mask = torch.tensor([[0, 1, 0], [0, 1, 2]])
        self._ablation_test_assert(
            net,
            net.model.linear1,
            (inp1, inp2, inp3),
            ([[316.0, 992.0, 316.0], [84.0, 632.0, 120.0]],),
            additional_input=(1,),
            baselines=baseline,
            ablations_per_eval=(1, 2, 3),
            layer_mask=layer_mask,
            attribute_to_layer_input=True,
        )
        self._ablation_test_assert(
            net,
            net.model.linear0,
            (inp1, inp2, inp3),
            [[316.0, 992.0, 316.0], [84.0, 632.0, 120.0]],
            additional_input=(1,),
            baselines=baseline,
            layer_mask=layer_mask,
            ablations_per_eval=(1, 2, 3),
        )

    def test_simple_multi_input_conv_intermediate(self):
        net = BasicModel_ConvNet_One_Conv(inplace=True)
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        self._ablation_test_assert(
            net,
            net.relu1,
            (inp, inp2),
            [[[4.0, 13.0], [40.0, 49.0]], [[0, 0], [-15.0, -24.0]]],
            ablations_per_eval=(1, 2, 4, 8, 12, 16),
        )
        self._ablation_test_assert(
            net,
            net.relu1,
            (inp, inp2),
            ([[[4.0, 13.0], [40.0, 49.0]], [[0, 0], [-15.0, -24.0]]],),
            baselines=torch.tensor(
                [[[-4.0, -13.0], [-2.0, -2.0]], [[0, 0], [0.0, 0.0]]]
            ),
            ablations_per_eval=(1, 2, 4, 8, 12, 16),
            attribute_to_layer_input=True,
        )
        self._ablation_test_assert(
            net,
            net.relu1,
            (inp, inp2),
            [[[17.0, 17.0], [67.0, 67.0]], [[0, 0], [-39.0, -39.0]]],
            ablations_per_eval=(1, 2, 4),
            layer_mask=torch.tensor([[[[0, 0], [1, 1]], [[2, 2], [3, 3]]]]),
        )

    def test_simple_multi_output_ablation(self):
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])
        self._ablation_test_assert(
            net, net.relu, inp, ([0.0, 7.0, 7.0, 7.0], [0.0, 7.0, 7.0, 7.0])
        )

    def test_simple_multi_output_input_ablation(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])
        self._ablation_test_assert(
            net,
            net.relu,
            inp,
            ([0.0, 7.0, 7.0, 7.0], [0.0, 7.0, 7.0, 7.0]),
            attribute_to_layer_input=True,
        )

    def _ablation_test_assert(
        self,
        model,
        layer,
        test_input,
        expected_ablation,
        layer_mask=None,
        additional_input=None,
        ablations_per_eval=(1,),
        baselines=None,
        target=0,
        attribute_to_layer_input=False,
    ):
        for batch_size in ablations_per_eval:
            ablation = LayerFeatureAblation(model, layer)
            attributions = ablation.attribute(
                test_input,
                target=target,
                layer_mask=layer_mask,
                additional_forward_args=additional_input,
                layer_baselines=baselines,
                ablations_per_eval=batch_size,
                attribute_to_layer_input=attribute_to_layer_input,
            )
            assertTensorTuplesAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
