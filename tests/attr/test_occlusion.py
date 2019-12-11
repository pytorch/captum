#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.occlusion import Occlusion

from .helpers.basic_models import (
    BasicModel3,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertTensorAlmostEqual, BaseTest


class Test(BaseTest):
    def test_improper_window_shape(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        occ = Occlusion(net)
        # Check error when too few sliding window dimensions
        with self.assertRaises(AssertionError):
            _ = occ.attribute(inp, sliding_window_shapes=((1, 2),), target=0,)

        # Check error when too many sliding window dimensions
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                (inp, inp), sliding_window_shapes=((1, 1, 2), (1, 1, 1, 2)), target=0,
            )

        # Check error when too many sliding window tuples
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                (inp, inp),
                sliding_window_shapes=((1, 1, 2), (1, 1, 2), (1, 1, 2)),
                target=0,
            )

    def test_improper_stride(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        occ = Occlusion(net)
        # Check error when too few stride dimensions
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                inp, sliding_window_shapes=(1, 2, 2), strides=(1, 2), target=0,
            )

        # Check error when too many stride dimensions
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                (inp, inp),
                sliding_window_shapes=((1, 1, 2), (1, 2, 2)),
                strides=((1, 1, 2), (2, 1, 2, 2)),
                target=0,
            )

        # Check error when too many stride tuples
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                (inp, inp),
                sliding_window_shapes=((1, 1, 2), (1, 2, 2)),
                strides=((1, 1, 2), (1, 2, 2), (1, 2, 2)),
                target=0,
            )

    def test_too_large_stride(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        occ = Occlusion(net)
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                inp, sliding_window_shapes=((1, 1, 2),), strides=2, target=0,
            )
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                (inp, inp),
                sliding_window_shapes=((1, 1, 2), (1, 4, 2)),
                strides=(2, (1, 2, 3)),
                target=0,
            )
        with self.assertRaises(AssertionError):
            _ = occ.attribute(
                inp, sliding_window_shapes=((2, 1, 2),), strides=2, target=0,
            )

    def test_simple_input(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._occlusion_test_assert(
            net,
            inp,
            [80.0, 200.0, 120.0],
            ablations_per_eval=(1, 2, 3),
            sliding_window_shapes=((1,)),
        )

    def test_simple_multi_input(self):
        net = BasicModel3()
        inp1 = torch.tensor([[-10.0], [3.0]])
        inp2 = torch.tensor([[-5.0], [1.0]])
        self._occlusion_test_assert(
            net,
            (inp1, inp2),
            ([0.0, 1.0], [0.0, -1.0]),
            sliding_window_shapes=((1,), (1,)),
        )

    def test_simple_multi_input_0d(self):
        net = BasicModel3()
        inp1 = torch.tensor([-10.0, 3.0])
        inp2 = torch.tensor([-5.0, 1.0])
        self._occlusion_test_assert(
            net,
            (inp1, inp2),
            ([0.0, 1.0], [0.0, -1.0]),
            sliding_window_shapes=((), ()),
            target=None,
        )

    def test_simple_input_larger_shape(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._occlusion_test_assert(
            net,
            inp,
            [200.0, 220.0, 240.0],
            ablations_per_eval=(1, 2, 3),
            sliding_window_shapes=((2,)),
            baselines=torch.tensor([10.0, 10.0, 10.0]),
        )

    def test_simple_input_shape_with_stride(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._occlusion_test_assert(
            net,
            inp,
            [280.0, 280.0, 120.0],
            ablations_per_eval=(1, 2, 3),
            sliding_window_shapes=((2,)),
            strides=2,
        )

    def test_multi_sample_ablation(self):
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._occlusion_test_assert(
            net,
            inp,
            [[8.0, 35.0, 12.0], [80.0, 200.0, 120.0]],
            ablations_per_eval=(1, 2, 3),
            sliding_window_shapes=((1,),),
        )

    def test_multi_input_ablation_with_mask(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        expected = (
            [[492.0, 492.0, 492.0], [400.0, 400.0, 400.0]],
            [[80.0, 200.0, 120.0], [0.0, 400.0, 0.0]],
            [[400.0, 420.0, 440.0], [48.0, 50.0, 52.0]],
        )
        self._occlusion_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            sliding_window_shapes=((3,), (1,), (2,)),
        )
        self._occlusion_test_assert(
            net,
            (inp1, inp2),
            expected[0:1],
            additional_input=(inp3, 1),
            ablations_per_eval=(1, 2, 3),
            sliding_window_shapes=((3,), (1,)),
        )

    def test_multi_input_ablation_with_baselines(self):
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        expected = (
            [[444.0, 444.0, 444.0], [328.0, 328.0, 328.0]],
            [[68.0, 188.0, 108.0], [-12.0, 388.0, -12.0]],
            [[368.0, 368.0, 24.0], [0.0, 0.0, -12.0]],
        )
        self._occlusion_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            baselines=(
                torch.tensor([[1.0, 4, 7], [3.0, 6, 9]]),
                3.0,
                torch.tensor([[4.0], [6]]),
            ),
            additional_input=(1,),
            sliding_window_shapes=((3,), (1,), (2,)),
            strides=(2, 1, 2),
        )

    def test_simple_multi_input_conv(self):
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).type(torch.FloatTensor)
        inp2 = torch.ones((1, 1, 4, 4))
        self._occlusion_test_assert(
            net,
            (inp, inp2),
            (67 * torch.ones_like(inp), 13 * torch.ones_like(inp2)),
            ablations_per_eval=(1, 2, 4, 8, 12, 16),
            sliding_window_shapes=((1, 4, 4), (1, 4, 4)),
        )
        self._occlusion_test_assert(
            net,
            (inp, inp2),
            (
                [
                    [17.0, 17.0, 17.0, 17.0],
                    [17.0, 17.0, 17.0, 17.0],
                    [64.0, 65.5, 65.5, 67.0],
                    [64.0, 65.5, 65.5, 67.0],
                ],
                [
                    [3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ),
            ablations_per_eval=(1, 3, 7, 14),
            sliding_window_shapes=((1, 2, 3), (1, 1, 2)),
            strides=((1, 2, 1), (1, 1, 2)),
        )

    def _occlusion_test_assert(
        self,
        model,
        test_input,
        expected_ablation,
        sliding_window_shapes=None,
        target=0,
        additional_input=None,
        ablations_per_eval=(1,),
        baselines=None,
        strides=None,
    ):
        for batch_size in ablations_per_eval:
            ablation = Occlusion(model)
            attributions = ablation.attribute(
                test_input,
                sliding_window_shapes=sliding_window_shapes,
                target=target,
                additional_forward_args=additional_input,
                baselines=baselines,
                ablations_per_eval=batch_size,
                strides=strides,
            )
            if isinstance(expected_ablation, tuple):
                for i in range(len(expected_ablation)):
                    assertTensorAlmostEqual(self, attributions[i], expected_ablation[i])
            else:
                assertTensorAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
