#!/usr/bin/env python3

from __future__ import print_function

import torch

from ..helpers.utils import (
    BaseTest,
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    assertArraysAlmostEqual,
    assert_delta,
)
from ..helpers.basic_models import (
    ReLULinearDeepLiftModel,
    BasicModel_MultiLayer,
    LinearMaxPoolLinearModel,
)

from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap


class TestDeepLift(BaseTest):
    def test_relu_layer_deeplift(self):
        model = ReLULinearDeepLiftModel(inplace=True)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.relu)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 15.0]])
        assert_delta(self, delta)

    def test_relu_layer_deeplift_multiple_output(self):
        model = BasicModel_MultiLayer(multi_input_module=True)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.relu)
        attributions, delta = layer_dl.attribute(
            inputs[0],
            baselines[0],
            target=0,
            attribute_to_layer_input=False,
            return_convergence_delta=True,
        )
        assertTensorTuplesAlmostEqual(
            self, attributions, ([[0.0, -1.0, -1.0, -1.0]], [[0.0, -1.0, -1.0, -1.0]])
        )
        assert_delta(self, delta)

    def test_relu_layer_deeplift_add_args(self):
        model = ReLULinearDeepLiftModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.relu)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            additional_forward_args=3.0,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 45.0]])
        assert_delta(self, delta)

    def test_linear_layer_deeplift(self):
        model = ReLULinearDeepLiftModel(inplace=True)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.l3)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 15.0]])
        assert_delta(self, delta)

    def test_relu_deeplift_with_custom_attr_func(self):
        model = ReLULinearDeepLiftModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()
        attr_method = LayerDeepLift(model, model.l3)
        self._relu_custom_attr_func_assert(attr_method, inputs, baselines, [[2.0]])

    def test_linear_layer_deeplift_batch(self):
        model = ReLULinearDeepLiftModel(inplace=True)
        _, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()
        x1 = torch.tensor(
            [[-10.0, 1.0, -5.0], [-10.0, 1.0, -5.0], [-10.0, 1.0, -5.0]],
            requires_grad=True,
        )
        x2 = torch.tensor(
            [[3.0, 3.0, 1.0], [3.0, 3.0, 1.0], [3.0, 3.0, 1.0]], requires_grad=True
        )
        inputs = (x1, x2)

        layer_dl = LayerDeepLift(model, model.l3)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 15.0]])
        assert_delta(self, delta)

        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=False,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions, [[15.0]])
        assert_delta(self, delta)

    def test_relu_layer_deepliftshap(self):
        model = ReLULinearDeepLiftModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        layer_dl_shap = LayerDeepLiftShap(model, model.relu)
        attributions, delta = layer_dl_shap.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 15.0]])
        assert_delta(self, delta)

    def test_relu_layer_deepliftshap_multiple_output(self):
        model = BasicModel_MultiLayer(multi_input_module=True)
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()

        layer_dl = LayerDeepLiftShap(model, model.relu)
        attributions, delta = layer_dl.attribute(
            inputs[0],
            baselines[0],
            target=0,
            attribute_to_layer_input=False,
            return_convergence_delta=True,
        )
        assertTensorTuplesAlmostEqual(
            self, attributions, ([[0.0, -1.0, -1.0, -1.0]], [[0.0, -1.0, -1.0, -1.0]])
        )
        assert_delta(self, delta)

    def test_linear_layer_deepliftshap(self):
        model = ReLULinearDeepLiftModel(inplace=True)
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        layer_dl_shap = LayerDeepLiftShap(model, model.l3)
        attributions, delta = layer_dl_shap.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 15.0]])
        assert_delta(self, delta)
        attributions, delta = layer_dl_shap.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=False,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions, [[15.0]])
        assert_delta(self, delta)

    def test_relu_deepliftshap_with_custom_attr_func(self):
        model = ReLULinearDeepLiftModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        attr_method = LayerDeepLiftShap(model, model.l3)
        self._relu_custom_attr_func_assert(
            attr_method, inputs, baselines, [[2.0], [2.0]]
        )

    def test_lin_maxpool_lin_classification(self):
        inputs = torch.ones(2, 4)
        baselines = torch.tensor([[1, 2, 3, 9], [4, 8, 6, 7]]).float()

        model = LinearMaxPoolLinearModel()
        dl = LayerDeepLift(model, model.pool1)
        attrs, delta = dl.attribute(
            inputs, baselines, target=0, return_convergence_delta=True
        )
        expected = [[[-8.0]], [[-7.0]]]
        expected_delta = [0.0, 0.0]
        assertArraysAlmostEqual(attrs.detach().numpy(), expected)
        assertArraysAlmostEqual(delta.detach().numpy(), expected_delta)

    def _relu_custom_attr_func_assert(self, attr_method, inputs, baselines, expected):
        def custom_attr_func(multipliers, inputs, baselines):
            return tuple(multiplier * 2 for multiplier in multipliers)

        attr = attr_method.attribute(
            inputs,
            baselines,
            custom_attribution_func=custom_attr_func,
            return_convergence_delta=True,
        )

        assertTensorAlmostEqual(self, attr[0], expected, 1e-19)


def _create_inps_and_base_for_deeplift_neuron_layer_testing():
    x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

    b1 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    b2 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    inputs = (x1, x2)
    baselines = (b1, b2)

    return inputs, baselines


def _create_inps_and_base_for_deepliftshap_neuron_layer_testing():
    x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

    b1 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
    )
    b2 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
    )

    inputs = (x1, x2)
    baselines = (b1, b2)

    return inputs, baselines
