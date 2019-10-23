#!/usr/bin/env python3
from __future__ import print_function

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _zeros

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


class Test(BaseTest):
    def test_multivariable_vanilla(self):
        self._assert_multi_variable("vanilla")

    def test_multivariable_smoothgrad(self):
        self._assert_multi_variable("smoothgrad")

    def test_multivariable_smoothgrad_sq(self):
        self._assert_multi_variable("smoothgrad_sq")

    def test_multivariable_vargrad(self):
        self._assert_multi_variable("vargrad")

    def test_multi_argument_vanilla(self):
        self._assert_multi_argument("vanilla")

    def test_multi_argument_smoothgrad(self):
        self._assert_multi_argument("smoothgrad")

    def test_multi_argument_smoothgrad_sq(self):
        self._assert_multi_argument("smoothgrad_sq")

    def test_multi_argument_vargrad(self):
        self._assert_multi_argument("vargrad")

    def test_univariable_vanilla(self):
        self._assert_univariable("vanilla")

    def test_univariable_smoothgrad(self):
        self._assert_univariable("smoothgrad")

    def test_univariable_smoothgrad_sq(self):
        self._assert_univariable("smoothgrad_sq")

    def test_univariable_vargrad(self):
        self._assert_univariable("vargrad")

    def test_multi_tensor_input_vanilla(self):
        self._assert_multi_tensor_input("vanilla")

    def test_multi_tensor_input_smoothgrad(self):
        self._assert_multi_tensor_input("smoothgrad")

    def test_multi_tensor_input_smoothgrad_sq(self):
        self._assert_multi_tensor_input("smoothgrad_sq")

    def test_multi_tensor_input_vargrad(self):
        self._assert_multi_tensor_input("vargrad")

    def test_batched_input_vanilla(self):
        self._assert_batched_tensor_input("vanilla")

    def test_batched_input_smoothgrad(self):
        self._assert_batched_tensor_input("smoothgrad")

    def test_batched_input_smoothgrad_sq(self):
        self._assert_batched_tensor_input("smoothgrad_sq")

    def test_batched_input_vargrad(self):
        self._assert_batched_tensor_input("vargrad")

    def test_batched_multi_input_vanilla(self):
        self._assert_batched_tensor_multi_input("vanilla")

    def test_batched_multi_input_smoothgrad(self):
        self._assert_batched_tensor_multi_input("smoothgrad")

    def test_batched_multi_input_smoothgrad_sq(self):
        self._assert_batched_tensor_multi_input("smoothgrad_sq")

    def test_batched_multi_input_vargrad(self):
        self._assert_batched_tensor_multi_input("vargrad")

    def _assert_multi_variable(self, type):
        model = BasicModel2()

        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0], requires_grad=True)

        baseline1 = torch.tensor([0.0])
        baseline2 = torch.tensor([0.0])

        attributions1 = self._compute_attribution_and_evaluate(
            model, (input1, input2), (baseline1, baseline2), type=type
        )
        if type == "vanilla":
            assertArraysAlmostEqual(attributions1[0].tolist(), [1.5], delta=0.05)
            assertArraysAlmostEqual(attributions1[1].tolist(), [-0.5], delta=0.05)
        model = BasicModel3()
        attributions2 = self._compute_attribution_and_evaluate(
            model, (input1, input2), (baseline1, baseline2), type=type
        )
        if type == "vanilla":
            assertArraysAlmostEqual(attributions2[0].tolist(), [1.5], delta=0.05)
            assertArraysAlmostEqual(attributions2[1].tolist(), [-0.5], delta=0.05)
            # Verifies implementation invariance
            self.assertEqual(
                sum(attribution for attribution in attributions1),
                sum(attribution for attribution in attributions2),
            )

    def _assert_univariable(self, type):
        model = BasicModel()
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
        )
        self._compute_attribution_and_evaluate(
            model, torch.tensor([0.0]), torch.tensor([0.0]), type=type
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([-1.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
        )

    def _assert_multi_argument(self, type):
        model = BasicModel4_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2]], requires_grad=True),
            ),
            baselines=(torch.zeros((1, 3)), torch.zeros((1, 3))),
            additional_forward_args=torch.arange(1.0, 4.0).reshape(1, 3),
            type=type,
        )
        # uses batching with an integer variable and nd-tensors as
        # additional forward arguments
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((2, 3)), torch.zeros((2, 3))),
            additional_forward_args=(torch.arange(1.0, 7.0).reshape(2, 3), 1),
            type=type,
        )
        # uses batching with an integer variable and python list
        # as additional forward arguments
        model = BasicModel5_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((2, 3)), torch.zeros((2, 3))),
            additional_forward_args=([2, 3], 1),
            type=type,
        )

    def _assert_multi_tensor_input(self, type):
        model = BasicModel6_MultiTensor()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 3.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 2.2]], requires_grad=True),
            ),
            type=type,
        )

    def _assert_batched_tensor_input(self, type):
        model = BasicModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.0, 1.3], [0.5, 0.1, 2.3], [1.5, 2.0, 1.3]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(model, input, type=type, target=0)
        self._compute_attribution_batch_helper_evaluate(model, input, target=0)

    def _assert_batched_tensor_multi_input(self, type):
        model = BasicModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.1, 1.9], [0.5, 0.0, 0.7], [1.5, 2.1, 1.1]], requires_grad=True
            ),
            torch.tensor(
                [[0.3, 1.9, 2.4], [0.5, 0.6, 2.1], [1.2, 2.1, 0.2]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(model, input, type=type, target=0)
        self._compute_attribution_batch_helper_evaluate(model, input, target=0)

    def _compute_attribution_and_evaluate(
        self,
        model,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        type="vanilla",
    ):
        r"""
            attrib_type: 'vanilla', 'smoothgrad', 'smoothgrad_sq', 'vargrad'
        """
        ig = IntegratedGradients(model)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)

        if baselines is None:
            baselines = _zeros(inputs)

        for method in [
            "riemann_right",
            "riemann_left",
            "riemann_middle",
            "riemann_trapezoid",
            "gausslegendre",
        ]:
            if type == "vanilla":
                attributions, delta = ig.attribute(
                    inputs,
                    baselines,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=2000,
                    target=target,
                    return_convergence_delta=True,
                )
                model.zero_grad()
                attributions_without_delta, delta = ig.attribute(
                    inputs,
                    baselines,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=2000,
                    target=target,
                    return_convergence_delta=True,
                )
                model.zero_grad()
                self.assertEqual([inputs[0].shape[0]], list(delta.shape))
                delta_external = ig.compute_convergence_delta(
                    attributions,
                    baselines,
                    inputs,
                    target=target,
                    additional_forward_args=additional_forward_args,
                )
                assertArraysAlmostEqual(delta, delta_external, 0.0)
            else:
                nt = NoiseTunnel(ig)
                n_samples = 5
                attributions, delta = nt.attribute(
                    inputs,
                    nt_type=type,
                    n_samples=n_samples,
                    stdevs=0.00000002,
                    baselines=baselines,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=2000,
                    return_convergence_delta=True,
                )
                attributions_without_delta = nt.attribute(
                    inputs,
                    nt_type=type,
                    n_samples=n_samples,
                    stdevs=0.00000002,
                    baselines=baselines,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=2000,
                )
                self.assertEqual([inputs[0].shape[0] * n_samples], list(delta.shape))

            for input, attribution in zip(inputs, attributions):
                self.assertEqual(attribution.shape, input.shape)
            self.assertTrue(all(abs(delta.numpy().flatten()) < 0.05))

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
        self, model, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        ig = IntegratedGradients(model)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)

        if baselines is None:
            baselines = _zeros(inputs)

        for method in [
            "riemann_right",
            "riemann_left",
            "riemann_middle",
            "riemann_trapezoid",
            "gausslegendre",
        ]:
            for internal_batch_size in [None, 1, 20]:
                attributions, delta = ig.attribute(
                    inputs,
                    baselines,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=200,
                    target=target,
                    internal_batch_size=internal_batch_size,
                    return_convergence_delta=True,
                )
                total_delta = 0
                for i in range(inputs[0].shape[0]):
                    attributions_indiv, delta_indiv = ig.attribute(
                        tuple(input[i : i + 1] for input in inputs),
                        tuple(baseline[i : i + 1] for baseline in baselines),
                        additional_forward_args=additional_forward_args,
                        method=method,
                        n_steps=200,
                        target=target,
                        return_convergence_delta=True,
                    )
                    total_delta += abs(delta_indiv).sum().item()
                    for j in range(len(attributions)):
                        assertArraysAlmostEqual(
                            attributions[j][i : i + 1].squeeze(0).tolist(),
                            attributions_indiv[j].squeeze(0).tolist(),
                        )
                self.assertAlmostEqual(
                    abs(delta).sum().item(), total_delta, delta=0.005
                )


if __name__ == "__main__":
    unittest.main()
