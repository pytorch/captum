from __future__ import print_function

import torch

from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.integrated_gradients import IntegratedGradients

from .helpers.utils import assertAttributionComparision, BaseTest
from .helpers.classification_models import SigmoidDeepLiftModel
from .helpers.classification_models import SoftmaxDeepLiftModel


class Test(BaseTest):
    def test_sigmoid_classification(self):
        num_in = 20
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        baseline = 0 * input
        target = torch.tensor(0)
        # TODO add test cases for multiple different layers
        model = SigmoidDeepLiftModel(num_in, 5, 1)
        dl = DeepLift(model)
        model.zero_grad()
        attributions, delta = dl.attribute(input, baseline, target=target)
        self._assert_attributions(model, attributions, input, baseline, delta, target)

        # compare with integrated gradients
        ig = IntegratedGradients(model)
        attributions_ig, delta_ig = ig.attribute(input, baseline, target=target)
        assertAttributionComparision(self, (attributions,), (attributions_ig,))

    def test_softmax_classification_zero_baseline(self):
        num_in = 40
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        baselines = 0 * input

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baselines)

    def test_softmax_classificatio_batch_zero_baseline(self):
        num_in = 40
        input = torch.arange(0.0, num_in * 3.0, requires_grad=True).reshape(3, num_in)
        baselines = 0 * input

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baselines)

    def test_softmax_classificatio_multi_baseline(self):
        num_in = 40
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        baselines = torch.randn(5, 40)

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLiftShap(model)

        self.softmax_classification(model, dl, input, baselines)

    def test_softmax_classificatio_batch_multi_baseline(self):
        num_in = 40
        input = torch.arange(0.0, num_in * 2.0, requires_grad=True).reshape(2, num_in)
        baselines = torch.randn(5, 40)

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLiftShap(model)

        self.softmax_classification(model, dl, input, baselines)

    def softmax_classification(self, model, attr_method, input, baselines):
        target = torch.tensor(2)
        # TODO add test cases for multiple different layers
        model.zero_grad()
        attributions, delta = attr_method.attribute(
            input, baselines=baselines, target=target
        )
        self._assert_attributions(model, attributions, input, baselines, delta, target)

        target2 = torch.tensor(1)
        attributions, delta = attr_method.attribute(
            input, baselines=baselines, target=target2
        )

        self._assert_attributions(model, attributions, input, baselines, delta, target2)

    def _assert_attributions(
        self, model, attributions, inputs, baselines, delta, target=None
    ):
        self.assertEqual(inputs.shape, attributions.shape)
        self.assertTrue(
            delta < 0.02,
            "The sum of attribution values: {} is not "
            "nearly equal to the difference between the endpoint".format(delta),
        )
        # compare with integrated gradients
        if inputs.shape == baselines.shape:
            ig = IntegratedGradients(model)
            attributions_ig, delta_ig = ig.attribute(
                inputs, baselines=baselines, target=target
            )
            assertAttributionComparision(self, attributions, attributions_ig)
