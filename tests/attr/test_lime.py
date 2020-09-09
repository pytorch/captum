#!/usr/bin/env python3

import unittest
from typing import Any, Callable, Tuple, Union

import torch

from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.lime import Lime

from ..helpers.basic import BaseTest, assertTensorTuplesAlmostEqual
from ..helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)


class Test(BaseTest):
    def test_simple_lime(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [73.3716, 193.3349, 113.3349],
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
        )

    def test_simple_lime_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [271.0, 271.0, 111.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
        )

    def test_simple_lime_with_baselines(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]])
        self._lime_test_assert(
            net,
            inp,
            [244.0, 244.0, 100.0],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_batch_lime(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0], [10.0, 14.0, 4.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [[53.0323, 120.6903, 62.1903], [53.0323, 120.6903, 62.1903]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=800,
        )

    def test_simple_batch_lime_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0], [10.0, 14.0, 4.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [[159.0, 159.0, 79.0], [159.0, 79.0, 159.0]],
            feature_mask=torch.tensor([[0, 0, 1], [0, 1, 0]]),
            perturbations_per_eval=(1, 2, 3),
            n_samples=300,
        )

    def test_multi_input_lime_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0]])
        expected = (
            [[87, 0, 0]],
            [[75, 0, 195]],
            [[0, 395, 35]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=2000,
        )

    def test_multi_input_lime_with_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 0, 0]])
        expected = (
            [[251.0, 591.0, 251.0]],
            [[251.0, 591.0, 0.0]],
            [[251.0, 251.0, 251.0]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline = (
            [[180, 576.0, 180]],
            [[180, 576.0, -8.0]],
            [[180, 180, 180]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
        )

    def test_multi_input_batch_lime_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]])
        expected = (
            [[81.5, 95.5, 55.5], [81.5, 95.5, 55.5]],
            [[35.5, 195.5, 95.5], [35.5, 195.5, 95.5]],
            [[0, 215.0, 15.5], [0, 215.0, 15.5]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=13000,
        )

    def test_multi_input_batch_lime(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected = (
            [[838.3333, 838.3333, 838.3333], [162.3333, 838.3333, 162.3333]],
            [[162.3333, 838.3333, 74.8333], [162.3333, 838.3333, 74.8333]],
            [[162.3333, 838.3333, 74.8333], [162.3333, 162.3333, 162.3333]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline = (
            [[806, 806, 806], [114, 806.0, 114]],
            [[114, 806, 56], [114, 806.0, 56.0]],
            [[114, 806, 56], [114, 114, 114]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    # Remaining tests are for cases where forward function returns a scalar
    # as either a float, integer, 0d tensor or 1d tensor.
    def test_single_lime_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(lambda inp: torch.sum(net(inp)).item())

    def test_single_lime_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(lambda inp: torch.sum(net(inp)))

    def test_single_lime_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_single_lime_scalar_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def _single_input_scalar_lime_assert(self, func: Callable) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._lime_test_assert(
            func,
            inp,
            [[75.0, 75.0, 17.0]],
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def test_multi_inp_lime_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(lambda *inp: torch.sum(net(*inp)))

    def test_multi_inp_lime_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(
            lambda *inp: torch.sum(net(*inp)).reshape(1)
        )

    def test_multi_inp_lime_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(
            lambda *inp: int(torch.sum(net(*inp)).item())
        )

    def test_multi_inp_lime_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(lambda *inp: torch.sum(net(*inp)).item())

    def _multi_input_scalar_lime_assert(self, func: Callable) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [20.0, 10.0, 13.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2]])
        expected = (
            [[3850.6666, 3850.6666, 3850.6666]],
            [[305.5, 3850.6666, 410.1]],
            [[305.5, 3850.6666, 410.1]],
        )

        self._lime_test_assert(
            func,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            perturbations_per_eval=(1,),
            target=None,
            n_samples=1500,
        )

    def _lime_test_assert(
        self,
        model: Callable,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_attr,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: Union[None, int] = 0,
        n_samples: int = 100,
        delta: float = 1.0,
    ) -> None:
        for batch_size in perturbations_per_eval:
            lime = Lime(model)
            attributions = lime.attribute(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
                n_samples=n_samples,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_attr, delta=delta, mode="max"
            )


if __name__ == "__main__":
    unittest.main()
