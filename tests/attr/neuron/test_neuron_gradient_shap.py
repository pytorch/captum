#!/usr/bin/env python3
import torch

from ..helpers.utils import BaseTest

from ..helpers.utils import assertTensorAlmostEqual
from ..helpers.classification_models import SoftmaxModel
from ..helpers.basic_models import BasicModel_MultiLayer

from captum.attr._core.neuron.neuron_gradient_shap import NeuronGradientShap
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)


class Test(BaseTest):
    def test_basic_multilayer(self):
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, 20.0, 10.0]])
        baselines = torch.randn(2, 3)

        self._assert_attributions(model, model.linear1, inputs, baselines, 0)

    def test_classification(self):
        def custom_baseline_fn(inputs):
            num_in = inputs.shape[1]
            return torch.arange(0.0, num_in * 5.0).reshape(5, num_in)

        num_in = 40
        n_samples = 100

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()

        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        baselines = custom_baseline_fn

        self._assert_attributions(model, model.relu1, inputs, baselines, 1, n_samples)

    def _assert_attributions(
        self, model, layer, inputs, baselines, neuron_ind, n_samples=5
    ):
        ngs = NeuronGradientShap(model, layer)
        nig = NeuronIntegratedGradients(model, layer)
        attrs_gs = ngs.attribute(
            inputs, neuron_ind, baselines=baselines, n_samples=n_samples, stdevs=0.09
        )

        if callable(baselines):
            baselines = baselines(inputs)

        attrs_ig = []
        for baseline in baselines:
            attrs_ig.append(
                nig.attribute(inputs, neuron_ind, baselines=baseline.unsqueeze(0))
            )
        attrs_ig = torch.stack(attrs_ig, axis=0).mean(axis=0)
        assertTensorAlmostEqual(self, attrs_gs, attrs_ig, 0.5)
