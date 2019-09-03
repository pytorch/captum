from __future__ import print_function

import torch

from captum.attr._core.deep_lift import DeepLift

from .helpers.base_test import BaseTest
from .helpers.classification_models import SigmoidDeepLiftModel
from .helpers.classification_models import SoftmaxDeepLiftModel


class Test(BaseTest):
    def test_sigmoid_classification(self):
        num_in = 20
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        target = torch.tensor(0)
        # TODO add test cases for multiple different layers
        model = SigmoidDeepLiftModel(num_in, 5, 1)
        dl = DeepLift(model)
        model.zero_grad()
        attributions = dl.attribute(input, target=target)
        self._assert_attributions(model, attributions, input, 0 * input, target)

    def test_softmax_classification(self):
        num_in = 40
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        target = torch.tensor(5)
        # TODO add test cases for multiple different layers
        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        model.zero_grad()
        dl = DeepLift(model)

        attributions = dl.attribute(input, target=target)
        self._assert_attributions(model, attributions, input, 0 * input, target)

        target2 = torch.tensor(1)
        attributions = dl.attribute(input, target=target2)
        self._assert_attributions(model, attributions, input, 0 * input, target=target2)

    def _assert_attributions(self, model, attributions, inputs, baselines, target=None):
        with torch.no_grad():
            diffs = torch.abs(
                model(inputs)[:, target] - model(baselines)[:, target]
            ).item()
        self.assertAlmostEqual(diffs, torch.abs(attributions.sum()).item(), delta=0.005)
