from __future__ import print_function

import torch
import numpy as np

from torch import nn

from .helpers.utils import assertArraysAlmostEqual, BaseTest
from .helpers.classification_models import SoftmaxModel
from .helpers.basic_models import BasicModel2
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.integrated_gradients import IntegratedGradients


class Test(BaseTest):

    # This test reproduces some of the test cases from the original implementation
    # https://github.com/slundberg/shap/
    # explainers/test_gradient.py
    def test_basic_multi_input(self):
        batch_size = 10

        x1 = torch.ones(batch_size, 3)
        x2 = torch.ones(batch_size, 4)
        inputs = (x1, x2)

        batch_size_baselines = 20
        baselines = (
            torch.zeros(batch_size_baselines, 3),
            torch.zeros(batch_size_baselines, 4),
        )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(7, 1)

            def forward(self, x1, x2):
                return self.linear(torch.cat((x1, x2), dim=-1))

        model = Net()
        model.eval()
        model.zero_grad()

        np.random.seed(0)
        torch.manual_seed(0)
        gradient_shap = GradientShap(model)
        attributions, delta = gradient_shap.attribute((x1, x2), baselines)
        self._assert_attribution_delta(inputs, attributions, delta)
        # Compare with integrated gradients
        ig = IntegratedGradients(model)
        baselines = (torch.zeros(batch_size, 3), torch.zeros(batch_size, 4))
        attributions_ig, delta_ig = ig.attribute(inputs, baselines=baselines)
        self._assert_shap_ig_comparision(attributions, attributions)

    def test_classification(self):
        num_in = 40
        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        baselines = torch.arange(0.0, num_in * 4.0).reshape(4, num_in)
        target = torch.tensor(1)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()
        model.zero_grad()

        gradient_shap = GradientShap(model)
        attributions, delta = gradient_shap.attribute(
            inputs, baselines=baselines, target=target, n_samples=1000, stdevs=0.9
        )
        self._assert_attribution_delta((inputs,), (attributions,), delta)

        # Compare with integrated gradients
        ig = IntegratedGradients(model)
        baselines = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        attributions_ig, delta_ig = ig.attribute(
            inputs, baselines=baselines, target=target
        )
        self._assert_shap_ig_comparision((attributions,), (attributions_ig,))

    def test_basic_relu_multi_input(self):
        model = BasicModel2()

        input1 = torch.tensor([[3.0]])
        input2 = torch.tensor([[1.0]], requires_grad=True)

        baseline1 = torch.tensor([[0.0]])
        baseline2 = torch.tensor([[0.0]])
        inputs = (input1, input2)
        baselines = (baseline1, baseline2)

        gs = GradientShap(model)
        attributions, delta = gs.attribute(inputs, baselines=baselines, n_samples=30000)
        self._assert_attribution_delta(inputs, attributions, delta)

        ig = IntegratedGradients(model)
        attributions_ig, delta_ig = ig.attribute(inputs, baselines=baselines)
        self._assert_shap_ig_comparision(attributions, attributions_ig)

    def _assert_attribution_delta(self, inputs, attributions, delta):
        for input, attribution in zip(inputs, attributions):
            self.assertEqual(attribution.shape, input.shape)
        self.assertTrue(
            delta < 0.001,
            "Sum of SHAP values does"
            " not match the difference of endpoints. %f" % (delta),
        )

    def _assert_shap_ig_comparision(self, attributions1, attributions2):
        for attribution1, attribution2 in zip(attributions1, attributions2):
            for attr_row1, attr_row2 in zip(
                attribution1.detach().numpy(), attribution2.detach().numpy()
            ):
                assertArraysAlmostEqual(attr_row1, attr_row2, delta=0.005)
