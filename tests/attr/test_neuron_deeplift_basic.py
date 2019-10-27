from __future__ import print_function

import torch

import numpy as np

from .helpers.utils import BaseTest, assertArraysAlmostEqual
from .helpers.basic_models import ReLULinearDeepLiftModel

from captum.attr._core.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from .test_layer_deeplift_basic import (
    _create_inps_and_base_for_deeplift_neuron_layer_testing,
    _create_inps_and_base_for_deepliftshap_neuron_layer_testing,
)


class Test(BaseTest):
    def test_relu_neuron_deeplift(self):
        model = ReLULinearDeepLiftModel()

        x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

        inputs = (x1, x2)
        # baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        neuron_dl = NeuronDeepLift(model, model.relu)
        attributions = neuron_dl.attribute(inputs, 0, attribute_to_neuron_input=True)
        assertArraysAlmostEqual(
            np.array([[-30.0, 1.0, -0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[1].detach().numpy()
        )

        attributions = neuron_dl.attribute(inputs, 0, attribute_to_neuron_input=False)
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[1].detach().numpy()
        )

    def test_linear_neuron_deeplift(self):
        model = ReLULinearDeepLiftModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        neuron_dl = NeuronDeepLift(model, model.l3)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=True
        )
        assertArraysAlmostEqual(
            np.array([[-0.0, 0.0, -0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[1].detach().numpy()
        )

        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )

        assertArraysAlmostEqual(
            np.array([[-0.0, 0.0, -0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[6.0, 9.0, 0.0]]), attributions[1].detach().numpy()
        )

    def test_relu_neuron_deeplift_shap(self):
        model = ReLULinearDeepLiftModel()
        inputs, baselines = (
            _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        )

        neuron_dl = NeuronDeepLiftShap(model, model.relu)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=True
        )
        assertArraysAlmostEqual(
            np.array([[-30.0, 1.0, -0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[1].detach().numpy()
        )

        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[1].detach().numpy()
        )

    def test_linear_neuron_deeplift_shap(self):
        model = ReLULinearDeepLiftModel()
        inputs, baselines = (
            _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        )

        neuron_dl = NeuronDeepLiftShap(model, model.l3)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=True
        )
        assertArraysAlmostEqual(
            np.array([[-0.0, 0.0, -0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[0.0, 0.0, 0.0]]), attributions[1].detach().numpy()
        )

        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )

        assertArraysAlmostEqual(
            np.array([[-0.0, 0.0, -0.0]]), attributions[0].detach().numpy()
        )
        assertArraysAlmostEqual(
            np.array([[6.0, 9.0, 0.0]]), attributions[1].detach().numpy()
        )
