from __future__ import print_function

import torch

from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.integrated_gradients import IntegratedGradients

from .helpers.utils import (
    assertAttributionComparision,
    assertArraysAlmostEqual,
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
        self._deeplift_helper(model, DeepLift(model), inputs, baselines)

    def test_tanh_deeplift(self):
        x1 = torch.tensor([-1.0], requires_grad=True)
        x2 = torch.tensor([-2.0], requires_grad=True)

        b1 = torch.tensor([0.0], requires_grad=True)
        b2 = torch.tensor([0.0], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = TanhDeepLiftModel()
        self._deeplift_helper(model, DeepLift(model), inputs, baselines)


    def test_relu_deeplift_batch(self):
        x1 = torch.tensor([[1.0], [1.0], [1.0], [1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_helper(model, DeepLift(model), inputs, baselines)

    def test_relu_deeplift_batch_4D_input(self):
        x1 = torch.ones(4, 1, 1, 1)
        x2 = torch.tensor([[[[2.0]]]] * 4)

        b1 = torch.zeros(4, 1, 1, 1)
        b2 = torch.zeros(4, 1, 1, 1)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_helper(model, DeepLiftShap(model), inputs, baselines)

    def test_relu_deeplift_multi_ref(self):
        x1 = torch.tensor([[1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_helper(model, DeepLiftShap(model), inputs, baselines)

    def test_relu_linear_deeplift(self):
        model = ReLULinearDeepLiftModel()
        x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

        b1 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        # expected = [[[0.0, 0.0]], [[6.0, 2.0]]]
        self._deeplift_helper(model, DeepLift(model), inputs, baselines)

    def _deeplift_helper(self, model, attr_method, inputs, baselines):
        input_bsz = inputs[0].shape[0]
        baseline_bsz = baselines[0].shape[0]
        # Run attribution multiple times to make sure that it is working as
        # expected
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
            if inputs[0].shape == baselines[0].shape:
                # Compare with Integrated Gradients
                ig = IntegratedGradients(model)
                attributions_ig = ig.attribute(inputs, baselines)
                assertAttributionComparision(self, attributions, attributions_ig)
