from __future__ import print_function

import torch

from captum.attr._core.deep_lift import DeepLift, DeepLiftShap

from .helpers.utils import BaseTest
from .helpers.basic_models import ReLUDeepLiftModel
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

    def test_relu_deeplift_batch(self):
        x1 = torch.tensor([[1.0], [1.0], [1.0], [1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_helper(model, DeepLift(model), inputs, baselines)

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
        # Run attribution multiple times to make sure that it is working as
        # expected
        for _ in range(5):
            model.zero_grad()
            attributions, delta = attr_method.attribute(inputs, baselines)

            self.assertTrue(
                delta < 0.00001,
                "The sum of attribution values is not "
                "nearly equal to the difference between the endpoint",
            )
            for input, attribution in zip(inputs, attributions):
                self.assertEqual(input.shape, attribution.shape)
