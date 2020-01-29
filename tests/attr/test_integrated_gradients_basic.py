#!/usr/bin/env python3

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _zeros, _tensorize_baseline
from captum.attr._utils.typing import Tensor, TensorOrTupleOfTensors

from .helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel3,
    BasicModel4_MultiArgs,
    BasicModel5_MultiArgs,
    BasicModel6_MultiTensor,
    BasicModel_MultiLayer,
)
from .helpers.utils import assertArraysAlmostEqual, assertTensorAlmostEqual, BaseTest

import unittest
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


class Test(BaseTest):
    def test_multivariable_vanilla(self) -> None:
        self._assert_multi_variable("vanilla", "riemann_right")

    def test_multivariable_smoothgrad(self) -> None:
        self._assert_multi_variable("smoothgrad", "riemann_left")

    def test_multivariable_smoothgrad_sq(self) -> None:
        self._assert_multi_variable("smoothgrad_sq", "riemann_middle")

    def test_multivariable_vargrad(self) -> None:
        self._assert_multi_variable("vargrad", "riemann_trapezoid")

    def test_multi_argument_vanilla(self) -> None:
        self._assert_multi_argument("vanilla", "gausslegendre")

    def test_multi_argument_smoothgrad(self) -> None:
        self._assert_multi_argument("smoothgrad", "riemann_right")

    def test_multi_argument_smoothgrad_sq(self) -> None:
        self._assert_multi_argument("smoothgrad_sq", "riemann_left")

    def test_multi_argument_vargrad(self) -> None:
        self._assert_multi_argument("vargrad", "riemann_middle")

    def test_univariable_vanilla(self) -> None:
        self._assert_univariable("vanilla", "riemann_trapezoid")

    def test_univariable_smoothgrad(self) -> None:
        self._assert_univariable("smoothgrad", "gausslegendre")

    def test_univariable_smoothgrad_sq(self) -> None:
        self._assert_univariable("smoothgrad_sq", "riemann_right")

    def test_univariable_vargrad(self) -> None:
        self._assert_univariable("vargrad", "riemann_left")

    def test_multi_tensor_input_vanilla(self) -> None:
        self._assert_multi_tensor_input("vanilla", "riemann_middle")

    def test_multi_tensor_input_smoothgrad(self) -> None:
        self._assert_multi_tensor_input("smoothgrad", "riemann_trapezoid")

    def test_multi_tensor_input_smoothgrad_sq(self) -> None:
        self._assert_multi_tensor_input("smoothgrad_sq", "gausslegendre")

    def test_multi_tensor_input_vargrad(self) -> None:
        self._assert_multi_tensor_input("vargrad", "riemann_right")

    def test_batched_input_vanilla(self) -> None:
        self._assert_batched_tensor_input("vanilla", "riemann_left")

    def test_batched_input_smoothgrad(self) -> None:
        self._assert_batched_tensor_input("smoothgrad", "riemann_middle")

    def test_batched_input_smoothgrad_sq(self) -> None:
        self._assert_batched_tensor_input("smoothgrad_sq", "riemann_trapezoid")

    def test_batched_input_vargrad(self) -> None:
        self._assert_batched_tensor_input("vargrad", "gausslegendre")

    def test_batched_multi_input_vanilla(self) -> None:
        self._assert_batched_tensor_multi_input("vanilla", "riemann_right")

    def test_batched_multi_input_smoothgrad(self) -> None:
        self._assert_batched_tensor_multi_input("smoothgrad", "riemann_left")

    def test_batched_multi_input_smoothgrad_sq(self) -> None:
        self._assert_batched_tensor_multi_input("smoothgrad_sq", "riemann_middle")

    def test_batched_multi_input_vargrad(self) -> None:
        self._assert_batched_tensor_multi_input("vargrad", "riemann_trapezoid")

    def _assert_multi_variable(
        self, type: str, approximation_method: str = "gausslegendre"
    ) -> None:
        model: nn.Module = BasicModel2()

        input1: Tensor = torch.tensor([3.0])
        input2: Tensor = torch.tensor([1.0], requires_grad=True)

        baseline1: Tensor = torch.tensor([0.0])
        baseline2: Tensor = torch.tensor([0.0])

        attributions1: Tuple[Tensor, ...] = self._compute_attribution_and_evaluate(
            model,
            (input1, input2),
            (baseline1, baseline2),
            type=type,
            approximation_method=approximation_method,
        )
        if type == "vanilla":
            assertArraysAlmostEqual(attributions1[0].tolist(), [1.5], delta=0.05)
            assertArraysAlmostEqual(attributions1[1].tolist(), [-0.5], delta=0.05)
        model: nn.Module = BasicModel3()
        attributions2: Tuple[Tensor, ...] = self._compute_attribution_and_evaluate(
            model,
            (input1, input2),
            (baseline1, baseline2),
            type=type,
            approximation_method=approximation_method,
        )
        if type == "vanilla":
            assertArraysAlmostEqual(attributions2[0].tolist(), [1.5], delta=0.05)
            assertArraysAlmostEqual(attributions2[1].tolist(), [-0.5], delta=0.05)
            # Verifies implementation invariance
            self.assertEqual(
                sum(attribution for attribution in attributions1),
                sum(attribution for attribution in attributions2),
            )

    def _assert_univariable(self, type, approximation_method="gausslegendre") -> None:
        model: nn.Module = BasicModel()
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
            approximation_method=approximation_method,
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            type=type,
            approximation_method=approximation_method,
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([-1.0], requires_grad=True),
            0.00001,
            type=type,
            approximation_method=approximation_method,
        )

    def _assert_multi_argument(
        self, type, approximation_method="gausslegendre"
    ) -> None:
        model: nn.Module = BasicModel4_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2]], requires_grad=True),
            ),
            baselines=(0.0, torch.zeros((1, 3))),
            additional_forward_args=torch.arange(1.0, 4.0).reshape(1, 3),
            type=type,
            approximation_method=approximation_method,
        )
        # uses batching with an integer variable and nd-tensors as
        # additional forward arguments
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((2, 3)), 0.0),
            additional_forward_args=(torch.arange(1.0, 7.0).reshape(2, 3), 1),
            type=type,
            approximation_method=approximation_method,
        )
        # uses batching with an integer variable and python list
        # as additional forward arguments
        model: nn.Module = BasicModel5_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(0.0, 0.00001),
            additional_forward_args=([2, 3], 1),
            type=type,
            approximation_method=approximation_method,
        )
        # similar to previous case plus baseline consists of a tensor and
        # a single example
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((1, 3)), 0.00001),
            additional_forward_args=([2, 3], 1),
            type=type,
            approximation_method=approximation_method,
        )

    def _assert_multi_tensor_input(
        self, type, approximation_method="gausslegendre"
    ) -> None:
        model: nn.Module = BasicModel6_MultiTensor()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 3.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 2.2]], requires_grad=True),
            ),
            type=type,
            approximation_method=approximation_method,
        )

    def _assert_batched_tensor_input(
        self, type, approximation_method="gausslegendre"
    ) -> None:
        model: nn.Module = BasicModel_MultiLayer()
        input: Tuple[Tensor] = (
            torch.tensor(
                [[1.5, 2.0, 1.3], [0.5, 0.1, 2.3], [1.5, 2.0, 1.3]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(
            model, input, type=type, target=0, approximation_method=approximation_method
        )
        self._compute_attribution_batch_helper_evaluate(
            model, input, target=0, approximation_method=approximation_method
        )

    def _assert_batched_tensor_multi_input(
        self, type, approximation_method="gausslegendre"
    ) -> None:
        model: nn.Module = BasicModel_MultiLayer()
        input: Tuple[Tensor, ...] = (
            torch.tensor(
                [[1.5, 2.1, 1.9], [0.5, 0.0, 0.7], [1.5, 2.1, 1.1]], requires_grad=True
            ),
            torch.tensor(
                [[0.3, 1.9, 2.4], [0.5, 0.6, 2.1], [1.2, 2.1, 0.2]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(
            model, input, type=type, target=0, approximation_method=approximation_method
        )
        self._compute_attribution_batch_helper_evaluate(
            model, input, target=0, approximation_method=approximation_method
        )

    def _compute_attribution_and_evaluate(
        self,
        model: nn.Module,
        inputs: TensorOrTupleOfTensors,
        baselines: Optional[Any] = None,
        target: Optional[int] = None,
        additional_forward_args: Optional[Any] = None,
        type: Optional[str] = "vanilla",
        approximation_method: str = "gausslegendre",
    ) -> Tuple[Tensor, ...]:
        r"""
            attrib_type: 'vanilla', 'smoothgrad', 'smoothgrad_sq', 'vargrad'
        """
        ig = IntegratedGradients(model)
        inputs_tuple: Tuple[Tensor, ...]
        if isinstance(inputs, Tensor):
            inputs_tuple = (inputs,)
        else:
            inputs_tuple = inputs

        baselines_tuple: Tuple[Any, ...]
        if baselines is None:
            baselines_tuple = _tensorize_baseline(inputs_tuple, _zeros(inputs_tuple))
        elif not isinstance(baselines, tuple):
            baselines_tuple = (baselines,)
        else:
            baselines_tuple = baselines

        attributions: Tuple[Tensor, ...]
        attributions_returned: Any
        attributions_without_delta: Tuple[Tensor, ...]
        attributions_without_delta_returned: Any
        delta: Tensor
        if type == "vanilla":
            attributions_returned, delta = ig.attribute(
                inputs_tuple,
                baselines_tuple,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                target=target,
                return_convergence_delta=True,
            )
            model.zero_grad()
            attributions_without_delta_returned, delta = ig.attribute(
                inputs_tuple,
                baselines_tuple,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                target=target,
                return_convergence_delta=True,
            )
            model.zero_grad()
            attributions = attributions_returned
            attributions_without_delta = attributions_without_delta_returned
            self.assertEqual([inputs_tuple[0].shape[0]], list(delta.shape))
            delta_external = ig.compute_convergence_delta(
                attributions,
                baselines_tuple,
                inputs_tuple,
                target=target,
                additional_forward_args=additional_forward_args,
            )
            assertArraysAlmostEqual(delta, delta_external, 0.0)
        else:
            nt = NoiseTunnel(ig)
            n_samples = 5
            attributions_returned, delta = nt.attribute(
                inputs_tuple,
                nt_type=type,
                n_samples=n_samples,
                stdevs=0.00000002,
                baselines=baselines_tuple,
                target=target,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                return_convergence_delta=True,
            )
            attributions_without_delta_returned = nt.attribute(
                inputs_tuple,
                nt_type=type,
                n_samples=n_samples,
                stdevs=0.00000002,
                baselines=baselines_tuple,
                target=target,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
            )
            attributions = attributions_returned
            attributions_without_delta = attributions_without_delta_returned

            self.assertEqual([inputs_tuple[0].shape[0] * n_samples], list(delta.shape))

        for input, attribution in zip(inputs_tuple, attributions):
            self.assertEqual(attribution.shape, input.shape)
        self.assertTrue(all(abs(delta.numpy().flatten()) < 0.07))

        # compare attributions retrieved with and without
        # `return_convergence_delta` flag
        for attribution, attribution_without_delta in zip(
            attributions, attributions_without_delta
        ):
            assertTensorAlmostEqual(
                self, attribution, attribution_without_delta, delta=0.05
            )

        return attributions

    def _compute_attribution_batch_helper_evaluate(
        self,
        model: nn.Module,
        inputs: TensorOrTupleOfTensors,
        baselines: Optional[Any] = None,
        target: Optional[int] = None,
        additional_forward_args: Optional[Any] = None,
        approximation_method: str = "gausslegendre",
    ):
        ig = IntegratedGradients(model)
        inputs_tuple: Tuple[Tensor, ...]
        if isinstance(inputs, Tensor):
            inputs_tuple = (inputs,)
        else:
            inputs_tuple = inputs

        baselines_tuple: Tuple[Any, ...]
        if baselines is None:
            baselines_tuple = _tensorize_baseline(inputs_tuple, _zeros(inputs_tuple))
        elif not isinstance(baselines, tuple):
            baselines_tuple = (baselines,)
        else:
            baselines_tuple = baselines

        for internal_batch_size in [None, 1, 20]:
            attributions: Tuple[Tensor, ...]
            attributions_returned: Any
            delta: Tensor
            attributions_returned, delta = ig.attribute(
                inputs_tuple,
                baselines_tuple,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=100,
                target=target,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )
            attributions = attributions_returned
            total_delta: float = 0.0
            for i in range(inputs_tuple[0].shape[0]):
                attributions_indiv: Tuple[Tensor, ...]
                attributions_indiv_returned: Any
                delta_indiv: Tensor
                attributions_indiv_returned, delta_indiv = ig.attribute(
                    tuple(input[i : i + 1] for input in inputs_tuple),
                    tuple(baseline[i : i + 1] for baseline in baselines_tuple),
                    additional_forward_args=additional_forward_args,
                    method=approximation_method,
                    n_steps=100,
                    target=target,
                    internal_batch_size=internal_batch_size,
                    return_convergence_delta=True,
                )
                attributions_indiv = attributions_indiv_returned
                total_delta += abs(delta_indiv).sum().item()
                for j in range(len(attributions)):
                    assertArraysAlmostEqual(
                        attributions[j][i : i + 1].squeeze(0).tolist(),
                        attributions_indiv[j].squeeze(0).tolist(),
                    )
            self.assertAlmostEqual(abs(delta).sum().item(), total_delta, delta=0.005)


if __name__ == "__main__":
    unittest.main()
