from __future__ import print_function

import torch


from captum.attr._core.deep_lift import DeepLift

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

        expected = [[2.0], [1.0]]
        model = ReLUDeepLiftModel()
        self._deeplift_helper(model, inputs, baselines, expected)

    def test_relu_deeplift_multi_ref(self):
        x1 = torch.tensor([[1.0]])
        x2 = torch.tensor([[2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.5], [0.0], [0.5]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.5], [0.0], [0.5]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        expected = [[2.0], [1.0]]
        model = ReLUDeepLiftModel()
        self._deeplift_helper(model, inputs, baselines, expected)

    def test_relu_linear_deeplift(self):
        model = ReLULinearDeepLiftModel()
        x1 = torch.tensor([[-10.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 1.0]], requires_grad=True)

        b1 = torch.tensor([[0.0, 0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0, 0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        # Run attribution multiple times to make sure that it is working as
        # expected
        expected = [[[0.0, 0.0]], [[6.0, 2.0]]]
        self._deeplift_helper(model, inputs, baselines, expected)

    def _deeplift_helper(self, model, inputs, baselines, expected):
        dl = DeepLift(model)
        # Run attribution multiple times to make sure that it is working as
        # expected
        for _ in range(5):
            model.zero_grad()
            attributions = dl.attribute(inputs, baselines)
            self.assertEqual(
                [attribution.detach().numpy().tolist() for attribution in attributions],
                expected,
            )
            self._assert_attributions(model, inputs, baselines, attributions)

    def _assert_attributions(self, model, inputs, baselines, attributions):
        diffs = [
            torch.abs(xout - bout).sum()
            for xout, bout in zip(model(*inputs), model(*baselines))
        ]
        self.assertAlmostEqual(
            sum(diffs), sum(attribution.sum() for attribution in attributions), 0.00001
        )
