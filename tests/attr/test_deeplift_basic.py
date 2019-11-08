#!/usr/bin/env python3

import torch

from inspect import signature

from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.integrated_gradients import IntegratedGradients

from .helpers.utils import (
    assertAttributionComparision,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
    BaseTest,
)
from .helpers.basic_models import ReLUDeepLiftModel, TanhDeepLiftModel
from .helpers.basic_models import ReLULinearDeepLiftModel


class Test(BaseTest):
    def test_relu_deeplift(self):
        x1 = torch.tensor([1.0], requires_grad=True)
        x2 = torch.tensor([2.0], requires_grad=True)

        b1 = torch.tensor([0.0], requires_grad=True)
        b2 = torch.tensor([0.0], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_tanh_deeplift(self):
        x1 = torch.tensor([-1.0], requires_grad=True)
        x2 = torch.tensor([-2.0], requires_grad=True)

        b1 = torch.tensor([0.0], requires_grad=True)
        b2 = torch.tensor([0.0], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = TanhDeepLiftModel()
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_deeplift_batch(self):
        x1 = torch.tensor([[1.0], [1.0], [1.0], [1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_linear_deeplift(self):
        model = ReLULinearDeepLiftModel()
        x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (0, 0.0001)

        # expected = [[[0.0, 0.0]], [[6.0, 2.0]]]
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_linear_deeplift_batch(self):
        model = ReLULinearDeepLiftModel()
        x1 = torch.tensor([[-10.0, 1.0, -5.0], [2.0, 3.0, 4.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0], [2.3, 5.0, 4.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (torch.zeros(1, 3), torch.rand(1, 3) * 0.001)
        # expected = [[[0.0, 0.0]], [[6.0, 2.0]]]
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_deepliftshap_batch_4D_input(self):
        x1 = torch.ones(4, 1, 1, 1)
        x2 = torch.tensor([[[[2.0]]]] * 4)

        b1 = torch.zeros(4, 1, 1, 1)
        b2 = torch.zeros(4, 1, 1, 1)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLiftShap(model), inputs, baselines)

    def test_relu_deepliftshap_multi_ref(self):
        x1 = torch.tensor([[1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLiftShap(model), inputs, baselines)

    def test_relu_deepliftshap_baselines_as_function(self):
        model = ReLULinearDeepLiftModel()
        x1 = torch.tensor([[-10.0, 1.0, -5.0]])
        x2 = torch.tensor([[3.0, 3.0, 1.0]])

        def gen_baselines():
            b1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            b2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            return (b1, b2)

        def gen_baselines_scalar():
            return (0.0, 0.0001)

        def gen_baselines_with_inputs(inputs):
            b1 = torch.cat([inputs[0], inputs[0] - 10])
            b2 = torch.cat([inputs[1], inputs[1] - 10])
            return (b1, b2)

        def gen_baselines_returns_array():
            b1 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            b2 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            return (b1, b2)

        inputs = (x1, x2)

        dl_shap = DeepLiftShap(model)
        self._deeplift_assert(model, dl_shap, inputs, gen_baselines)
        self._deeplift_assert(model, dl_shap, inputs, gen_baselines_with_inputs)
        with self.assertRaises(AssertionError):
            self._deeplift_assert(
                model, DeepLiftShap(model), inputs, gen_baselines_returns_array
            )
        with self.assertRaises(AssertionError):
            self._deeplift_assert(model, dl_shap, inputs, gen_baselines_scalar)

        baselines = gen_baselines()
        attributions = dl_shap.attribute(inputs, baselines)
        attributions_with_func = dl_shap.attribute(inputs, gen_baselines)
        assertTensorAlmostEqual(self, attributions[0], attributions_with_func[0])
        assertTensorAlmostEqual(self, attributions[1], attributions_with_func[1])

    def _deeplift_assert(self, model, attr_method, inputs, baselines):
        input_bsz = len(inputs[0])
        if callable(baselines):
            baseline_parameters = signature(baselines).parameters
            if len(baseline_parameters) > 0:
                baselines = baselines(inputs)
            else:
                baselines = baselines()

        baseline_bsz = (
            len(baselines[0]) if isinstance(baselines[0], torch.Tensor) else 1
        )
        # Run attribution multiple times to make sure that it is
        # working as expected
        for _ in range(5):
            model.zero_grad()
            attributions, delta = attr_method.attribute(
                inputs, baselines, return_convergence_delta=True
            )
            attributions_without_delta = attr_method.attribute(inputs, baselines)

            for attribution, attribution_without_delta in zip(
                attributions, attributions_without_delta
            ):
                self.assertTrue(
                    torch.all(torch.eq(attribution, attribution_without_delta))
                )

            if isinstance(attr_method, DeepLiftShap):
                self.assertEqual([input_bsz * baseline_bsz], list(delta.shape))
            else:
                self.assertEqual([input_bsz], list(delta.shape))
                delta_external = attr_method.compute_convergence_delta(
                    attributions, baselines, inputs
                )
                assertArraysAlmostEqual(delta, delta_external, 0.0)

            delta_condition = all(abs(delta.numpy().flatten()) < 0.00001)
            self.assertTrue(
                delta_condition,
                "The sum of attribution values {} is not "
                "nearly equal to the difference between the endpoint for "
                "some samples".format(delta),
            )
            for input, attribution in zip(inputs, attributions):
                self.assertEqual(input.shape, attribution.shape)
            if (
                isinstance(baselines[0], (int, float))
                or inputs[0].shape == baselines[0].shape
            ):
                # Compare with Integrated Gradients
                ig = IntegratedGradients(model)
                attributions_ig = ig.attribute(inputs, baselines)
                assertAttributionComparision(self, attributions, attributions_ig)
