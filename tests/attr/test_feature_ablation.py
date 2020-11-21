#!/usr/bin/env python3

import unittest
from typing import Any, Callable, List, Tuple, Union, cast

import torch
from torch import Tensor

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_ablation import FeatureAblation
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import (
    BasicModel,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModelWithSparseInputs,
)


class Test(BaseTest):
    r"""
    The following conversion tests are underlying assumptions
    made by the rest of tests in this file.

    We are testing them explicitly just in case they break behaviour
    in the future. As in this case it will be easier to update the tests.
    """

    def test_python_float_conversion(self) -> None:
        x = torch.tensor(3, dtype=cast(torch.dtype, float))
        self.assertEqual(x.dtype, torch.float64)

    def test_python_int_conversion(self) -> None:
        x = torch.tensor(5, dtype=cast(torch.dtype, int))
        self.assertEqual(x.dtype, torch.int64)

    def test_float32_tensor_item_conversion(self) -> None:
        x = torch.tensor(5, dtype=torch.float32)
        y = torch.tensor(x.item())  # .item() returns a python float

        # for whatever reason it is only
        # dtype == torch.float64 if you provide dtype=float
        self.assertEqual(y.dtype, torch.float32)

    def test_int32_tensor_item_conversion(self) -> None:
        x = torch.tensor(5, dtype=torch.int32)
        y = torch.tensor(x.item())  # .item() returns a python int
        self.assertEqual(y.dtype, torch.int64)

    def test_simple_ablation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net, inp, [[80.0, 200.0, 120.0]], perturbations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_int_to_int(self) -> None:
        net = BasicModel()
        inp = torch.tensor([[-3, 1, 2]])
        self._ablation_test_assert(
            net, inp, [[-3, 0, 0]], perturbations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_int_to_float(self) -> None:
        net = BasicModel()

        def wrapper_func(inp):
            return net(inp).float()

        inp = torch.tensor([[-3, 1, 2]])
        self._ablation_test_assert(
            wrapper_func, inp, [[-3.0, 0.0, 0.0]], perturbations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [[280.0, 280.0, 120.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_ablation_with_baselines(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [[248.0, 248.0, 104.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            inp,
            [[8.0, 35.0, 12.0], [80.0, 200.0, 120.0]],
            perturbations_per_eval=(1, 2, 3),
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
            perturbations_per_eval=(1, 2, 3),
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
            perturbations_per_eval=(1, 2, 3),
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
            perturbations_per_eval=(1, 2, 3),
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
            perturbations_per_eval=(1, 2, 3),
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
            perturbations_per_eval=(1, 2, 3),
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
            perturbations_per_eval=(1, 2, 4, 8, 12, 16),
        )
        self._ablation_test_assert(
            net,
            (inp, inp2),
            (
                [
                    [
                        [
                            [0.0, 2.0, 4.0, 3.0],
                            [4.0, 9.0, 10.0, 7.0],
                            [4.0, 13.0, 14.0, 11.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
                [
                    [
                        [
                            [1.0, 2.0, 2.0, 1.0],
                            [1.0, 2.0, 2.0, 1.0],
                            [1.0, 2.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
            ),
            perturbations_per_eval=(1, 3, 7, 14),
        )

    # Remaining tests are for cases where forward function returns a scalar
    # per batch, as either a float, integer, 0d tensor or 1d tensor.
    def test_error_perturbations_per_eval_limit_batch_scalar(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        ablation = FeatureAblation(lambda inp: torch.sum(net(inp)).item())
        with self.assertRaises(AssertionError):
            _ = ablation.attribute(inp, perturbations_per_eval=2)

    def test_error_agg_mode_arbitrary_output(self) -> None:
        net = BasicModel_MultiLayer()

        # output 3 numbers for the entire batch
        # note that the batch size == 2
        def forward_func(inp):
            pred = net(inp)
            return torch.stack([pred.sum(), pred.max(), pred.min()])

        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        ablation = FeatureAblation(forward_func)
        with self.assertRaises(AssertionError):
            _ = ablation.attribute(inp, perturbations_per_eval=2)

    def test_error_agg_mode_incorrect_fm(self) -> None:
        def forward_func(inp):
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[0, 1, 2], [0, 0, 1]])

        ablation = FeatureAblation(forward_func)
        with self.assertRaises(AssertionError):
            _ = ablation.attribute(inp, perturbations_per_eval=1, feature_mask=mask)

    def test_empty_sparse_features(self) -> None:
        model = BasicModelWithSparseInputs()
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        inp2 = torch.tensor([])
        exp: Tuple[List[List[float]], List[float]] = ([[9.0, -3.0, 12.0]], [0.0])
        self._ablation_test_assert(model, (inp1, inp2), exp, target=None)

    def test_sparse_features(self) -> None:
        model = BasicModelWithSparseInputs()
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        # Length of sparse index list may not match # of examples
        inp2 = torch.tensor([1, 7, 2, 4, 5, 3, 6])
        self._ablation_test_assert(
            model, (inp1, inp2), ([[9.0, -3.0, 12.0]], [2.0]), target=None
        )

    def test_single_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp)).item(), dtype=torch.float64
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
            lambda inp: int(torch.sum(net(inp)).item()), dtype=torch.int64
        )

    def test_multi_sample_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            lambda inp: torch.sum(net(inp)).item(),
            dtype=torch.float64,
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
            lambda inp: int(torch.sum(net(inp)).item()), dtype=torch.int64
        )

    def test_multi_inp_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_ablation_assert(
            lambda *inp: torch.sum(net(*inp)).item(),
            dtype=torch.float64,
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
            lambda *inp: int(torch.sum(net(*inp)).item()), dtype=torch.int64
        )

    def test_unassociated_output_3d_tensor(self) -> None:
        def forward_func(inp):
            return torch.ones(1, 5, 3, 2)

        inp = torch.randn(10, 5)
        mask = torch.arange(5).unsqueeze(0)
        self._ablation_test_assert(
            model=forward_func,
            test_input=inp,
            baselines=None,
            target=None,
            feature_mask=mask,
            perturbations_per_eval=(1,),
            expected_ablation=torch.zeros((5 * 3 * 2,) + inp[0].shape).numpy().tolist(),
        )

    def test_single_inp_ablation_multi_output_aggr(self) -> None:
        def forward_func(inp):
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[0, 1, 2]])
        self._ablation_test_assert(
            model=forward_func,
            test_input=inp,
            feature_mask=mask,
            baselines=None,
            target=None,
            perturbations_per_eval=(1,),
            # should just be the first input spread across each feature
            expected_ablation=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        )

    def test_single_inp_ablation_multi_output_aggr_mask_none(self) -> None:
        def forward_func(inp):
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._ablation_test_assert(
            model=forward_func,
            test_input=inp,
            feature_mask=None,
            baselines=None,
            target=None,
            perturbations_per_eval=(1,),
            # should just be the first input spread across each feature
            expected_ablation=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        )

    def test_single_inp_ablation_multi_output_aggr_non_standard(self) -> None:
        def forward_func(inp):
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[0, 0, 1]])
        self._ablation_test_assert(
            model=forward_func,
            test_input=inp,
            feature_mask=mask,
            baselines=None,
            target=None,
            perturbations_per_eval=(1,),
            expected_ablation=[[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        )

    def _single_input_one_sample_batch_scalar_ablation_assert(
        self, func: Callable, dtype: torch.dtype = torch.float32
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._ablation_test_assert(
            func,
            inp,
            torch.tensor([[82.0, 82.0, 24.0]], dtype=dtype),
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def _single_input_multi_sample_batch_scalar_ablation_assert(
        self,
        func: Callable,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._ablation_test_assert(
            func,
            inp,
            torch.tensor([[642.0, 642.0, 264.0]], dtype=dtype),
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def _multi_input_batch_scalar_ablation_assert(
        self, func: Callable, dtype: torch.dtype = torch.float32
    ) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2]])
        expected = (
            torch.tensor([[1784, 1784, 1784]], dtype=dtype),
            torch.tensor([[160, 1200, 240]], dtype=dtype),
            torch.tensor([[16, 880, 104]], dtype=dtype),
        )

        self._ablation_test_assert(
            func,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            perturbations_per_eval=(1,),
            target=None,
        )

    def _ablation_test_assert(
        self,
        model: Callable,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_ablation: Union[
            Tensor,
            Tuple[Tensor, ...],
            # NOTE: mypy doesn't support recursive types
            # would do a List[NestedList[Union[int, float]]
            # or Tuple[NestedList[Union[int, float]]
            # but... we can't.
            #
            # See https://github.com/python/mypy/issues/731
            List[Any],
            Tuple[List[Any], ...],
        ],
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: TargetType = 0,
    ) -> None:
        for batch_size in perturbations_per_eval:
            ablation = FeatureAblation(model)
            self.assertTrue(ablation.multiplies_by_inputs)
            attributions = ablation.attribute(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
            )
            if isinstance(expected_ablation, tuple):
                for i in range(len(expected_ablation)):
                    expected = expected_ablation[i]
                    if not isinstance(expected, torch.Tensor):
                        expected = torch.tensor(expected)

                    self.assertEqual(attributions[i].shape, expected.shape)
                    self.assertEqual(attributions[i].dtype, expected.dtype)
                    assertTensorAlmostEqual(self, attributions[i], expected)
            else:
                if not isinstance(expected_ablation, torch.Tensor):
                    expected_ablation = torch.tensor(expected_ablation)

                self.assertEqual(attributions.shape, expected_ablation.shape)
                self.assertEqual(attributions.dtype, expected_ablation.dtype)
                assertTensorAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
