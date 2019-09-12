from __future__ import print_function

import torch
import numpy as np

from torch import nn

from .helpers.utils import BaseTest
from .helpers.classification_models import SoftmaxModel
from captum.attr._core.gradient_shap import GradientShap


class Test(BaseTest):

    # This test reproduces some of the test cases from the original implementation
    # https://github.com/slundberg/shap/
    # explainers/test_gradient.py
    def test_basic_multi_input(self):
        batch_size = 10

        x1 = torch.ones(batch_size, 3)
        x2 = torch.ones(batch_size, 4)

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
        self.assertEqual(attributions[0].shape, x1.shape)
        self.assertEqual(attributions[1].shape, x2.shape)
        self.assertTrue(
            delta < 0.05,
            "Sum of shap values does"
            " not match the difference of endpoints. %f" % (delta),
        )

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
        self.assertEqual(attributions.shape, inputs.shape)
        self.assertTrue(
            delta < 0.05,
            "Sum of shap values does"
            " not match the difference of endpoints. %f" % (delta),
        )
