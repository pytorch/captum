#!/usr/bin/env python3

import unittest

import torch
from torch import Tensor
from captum.attr._core.feature_ablation import FeatureAblation
from typing import List, Tuple, Union, Optional, Any, Callable

from .helpers.basic_models import (
    BasicModel,
    BasicModelWithSparseInputs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from .helpers.utils import assertTensorAlmostEqual, BaseTest
from captum.attr._utils.typing import TensorOrTupleOfTensors


class Test(BaseTest):
    def test_simple_ablation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net, inp, [80.0, 200.0, 120.0], ablations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_int_to_int(self) -> None:
        net = BasicModel()
        inp = torch.tensor([[-3, 1, 2]])
        self._ablation_test_assert(
            net, inp, [-3.0, 0.0, 0.0], ablations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_int_to_float(self) -> None:
        net = BasicModel()

        def wrapper_func(inp):
            return net(inp).float()

        inp = torch.tensor([[-3, 1, 2]])
        self._ablation_test_assert(
            wrapper_func, inp, [-3.0, 0.0, 0.0], ablations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [280.0, 280.0, 120.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            ablations_per_eval=(1, 2, 3),
        )

    def test_simple_ablation_with_baselines(self) -> None:
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

    def test_multi_sample_ablation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [[8.0, 35.0, 12.0], [80.0, 200.0, 120.0]],
            ablations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation_with_mask(self) -> None:
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

    def test_multi_input_ablation_with_mask(self) -> None:
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
            feature_mask=(mask1, mask2),
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

    def test_multi_input_ablation(self) -> None:
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

    def test_simple_multi_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
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

    # Remaining tests are for cases where forward function returns a scalar
    # per batch, as either a float, integer, 0d tensor or 1d tensor.
    def test_error_ablations_per_eval_limit_batch_scalar(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        ablation = FeatureAblation(lambda inp: torch.sum(net(inp)).item())
        with self.assertRaises(AssertionError):
            _ = ablation.attribute(inp, ablations_per_eval=2)

    def test_empty_sparse_features(self) -> None:
        model = BasicModelWithSparseInputs()
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        inp2 = torch.tensor([])
        exp: Tuple[List[List[float]], ...] = (
            [[9.0, -3.0, 12.0]],
            [[]],
        )
        self._ablation_test_assert(
            model, (inp1, inp2), exp, target=None,
        )

    def test_sparse_features(self) -> None:
        model = BasicModelWithSparseInputs()
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        # Length of sparse index list may not match # of examples
        inp2 = torch.tensor([1, 7, 2, 4, 5, 3, 6])
        self._ablation_test_assert(
            model, (inp1, inp2), ([[9.0, -3.0, 12.0]], [[2.0]],), target=None,
        )

    def test_single_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp)).item()
        )

    def test_single_ablation_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp))
        )

    def test_single_ablation_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_single_ablation_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_ablation_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def test_multi_sample_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp)).item()
        )

    def test_multi_sample_ablation_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp))
        )

    def test_multi_sample_ablation_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_multi_sample_ablation_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def test_multi_inp_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_ablation_assert(
            lambda *inp: torch.sum(net(*inp)).item()
        )

    def test_multi_inp_ablation_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_ablation_assert(
            lambda *inp: torch.sum(net(*inp))
        )

    def test_multi_inp_ablation_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_ablation_assert(
            lambda *inp: torch.sum(net(*inp)).reshape(1)
        )

    def test_mutli_inp_ablation_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_ablation_assert(
            lambda *inp: int(torch.sum(net(*inp)).item())
        )

    def _single_input_one_sample_batch_scalar_ablation_assert(
        self, func: Callable
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._ablation_test_assert(
            func,
            inp,
            [[82.0, 82.0, 24.0]],
            feature_mask=mask,
            ablations_per_eval=(1,),
            target=None,
        )

    def _single_input_multi_sample_batch_scalar_ablation_assert(
        self, func: Callable
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._ablation_test_assert(
            func,
            inp,
            [[642.0, 642.0, 264.0]],
            feature_mask=mask,
            ablations_per_eval=(1,),
            target=None,
        )

    def _multi_input_batch_scalar_ablation_assert(self, func: Callable) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2]])
        expected = (
            [[1784.0, 1784.0, 1784.0]],
            [[160.0, 1200.0, 240.0]],
            [[16.0, 880.0, 104.0]],
        )

        self._ablation_test_assert(
            func,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            ablations_per_eval=(1,),
            target=None,
        )

    def _ablation_test_assert(
        self,
        model: Callable,
        test_input: TensorOrTupleOfTensors,
        expected_ablation: Union[
            List[float],
            List[List[float]],
            Tuple[List[List[float]], ...],
            Tuple[Tensor, ...],
        ],
        feature_mask: Optional[TensorOrTupleOfTensors] = None,
        additional_input: Any = None,
        ablations_per_eval: Tuple[int, ...] = (1,),
        baselines: Optional[
            Union[Tensor, int, float, Tuple[Union[Tensor, int, float], ...]]
        ] = None,
        target: Optional[
            Union[int, Tuple[int, ...], Tensor, List[Tuple[int, ...]]]
        ] = 0,
    ) -> None:
        for batch_size in ablations_per_eval:
            ablation = FeatureAblation(model)
            attributions = ablation.attribute(
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
