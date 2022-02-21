#!/usr/bin/env python3

from __future__ import print_function

from typing import List, Tuple, Union, cast

import torch
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap
from tests.helpers.basic import (
    BaseTest,
    assert_delta,
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
)
from tests.helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_ConvNet_MaxPool3d,
    BasicModel_MaxPool_ReLU,
    BasicModel_MultiLayer,
    LinearMaxPoolLinearModel,
    ReLULinearModel,
)
from torch import Tensor


class TestDeepLift(BaseTest):
    def test_relu_layer_deeplift(self) -> None:
        model = ReLULinearModel(inplace=False)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.relu)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [0.0, 15.0])
        assert_delta(self, delta)

    def test_relu_layer_deeplift_wo_mutliplying_by_inputs(self) -> None:
        model = ReLULinearModel(inplace=False)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.relu, multiply_by_inputs=False)
        attributions = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [0.0, 1.0])

    def test_relu_layer_deeplift_multiple_output(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.multi_relu)
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

    def test_relu_layer_deeplift_add_args(self) -> None:
        model = ReLULinearModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.relu)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            additional_forward_args=3.0,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [0.0, 45.0])
        assert_delta(self, delta)

    def test_linear_layer_deeplift(self) -> None:
        model = ReLULinearModel(inplace=False)
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        layer_dl = LayerDeepLift(model, model.l3)
        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [0.0, 15.0])
        assert_delta(self, delta)

    def test_relu_deeplift_with_custom_attr_func(self) -> None:
        model = ReLULinearModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()
        attr_method = LayerDeepLift(model, model.l3)
        self._relu_custom_attr_func_assert(attr_method, inputs, baselines, [[2.0]])

    def test_inplace_maxpool_relu_with_custom_attr_func(self) -> None:
        model = BasicModel_MaxPool_ReLU(inplace=False)
        inp = torch.tensor([[[1.0, 2.0, -4.0], [-3.0, -2.0, -1.0]]])
        dl = LayerDeepLift(model, model.maxpool)

        def custom_att_func(mult, inp, baseline):
            assertTensorAlmostEqual(self, mult[0], [[[1.0], [0.0]]])
            assertTensorAlmostEqual(self, inp[0], [[[2.0], [-1.0]]])
            assertTensorAlmostEqual(self, baseline[0], [[[0.0], [0.0]]])
            return mult

        dl.attribute(inp, custom_attribution_func=custom_att_func)

    def test_linear_layer_deeplift_batch(self) -> None:
        model = ReLULinearModel(inplace=False)
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
        assertTensorAlmostEqual(self, attributions[0], [0.0, 15.0])
        assert_delta(self, delta)

        attributions, delta = layer_dl.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=False,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions, [[15.0], [15.0], [15.0]])
        assert_delta(self, delta)

    def test_relu_layer_deepliftshap(self) -> None:
        model = ReLULinearModel()
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
        assertTensorAlmostEqual(self, attributions[0], [0.0, 15.0])
        assert_delta(self, delta)

    def test_relu_layer_deepliftshap_wo_mutliplying_by_inputs(self) -> None:
        model = ReLULinearModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        layer_dl_shap = LayerDeepLiftShap(model, model.relu, multiply_by_inputs=False)
        attributions = layer_dl_shap.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=True,
        )
        assertTensorAlmostEqual(self, attributions[0], [0.0, 1.0])

    def test_relu_layer_deepliftshap_multiple_output(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()

        layer_dl = LayerDeepLiftShap(model, model.multi_relu)
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

    def test_linear_layer_deepliftshap(self) -> None:
        model = ReLULinearModel(inplace=False)
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
        assertTensorAlmostEqual(self, attributions[0], [0.0, 15.0])
        assert_delta(self, delta)
        attributions, delta = layer_dl_shap.attribute(
            inputs,
            baselines,
            attribute_to_layer_input=False,
            return_convergence_delta=True,
        )
        assertTensorAlmostEqual(self, attributions, [[15.0]])
        assert_delta(self, delta)

    def test_relu_deepliftshap_with_custom_attr_func(self) -> None:
        model = ReLULinearModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        attr_method = LayerDeepLiftShap(model, model.l3)
        self._relu_custom_attr_func_assert(attr_method, inputs, baselines, [[2.0]])

    def test_lin_maxpool_lin_classification(self) -> None:
        inputs = torch.ones(2, 4)
        baselines = torch.tensor([[1, 2, 3, 9], [4, 8, 6, 7]]).float()

        model = LinearMaxPoolLinearModel()
        dl = LayerDeepLift(model, model.pool1)
        attrs, delta = dl.attribute(
            inputs, baselines, target=0, return_convergence_delta=True
        )
        expected = [[[-8.0]], [[-7.0]]]
        expected_delta = [0.0, 0.0]
        assertTensorAlmostEqual(self, cast(Tensor, attrs), expected, 0.0001, "max")
        assertTensorAlmostEqual(self, delta, expected_delta, 0.0001, "max")

    def test_convnet_maxpool2d_classification(self) -> None:
        inputs = 100 * torch.randn(2, 1, 10, 10)

        model = BasicModel_ConvNet()
        model.eval()

        dl = LayerDeepLift(model, model.pool1)
        dl2 = LayerDeepLift(model, model.conv2)

        attr = dl.attribute(inputs, target=0)
        attr2 = dl2.attribute(inputs, target=0, attribute_to_layer_input=True)

        self.assertTrue(cast(Tensor, attr).sum() == cast(Tensor, attr2).sum())

    def test_convnet_maxpool3d_classification(self) -> None:
        inputs = 100 * torch.randn(2, 1, 10, 10, 10)

        model = BasicModel_ConvNet_MaxPool3d()
        model.eval()

        dl = LayerDeepLift(model, model.pool1)
        dl2 = LayerDeepLift(model, model.conv2)
        # with self.assertRaises(AssertionError) doesn't run with Cicle CI
        # the error is being converted into RuntimeError

        attr = dl.attribute(inputs, target=0, attribute_to_layer_input=False)
        attr2 = dl2.attribute(inputs, target=0, attribute_to_layer_input=True)
        self.assertTrue(cast(Tensor, attr).sum() == cast(Tensor, attr2).sum())

    def _relu_custom_attr_func_assert(
        self,
        attr_method: Union[LayerDeepLift, LayerDeepLiftShap],
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Union[Tensor, Tuple[Tensor, ...]],
        expected: List[List[float]],
    ) -> None:
        def custom_attr_func(multipliers, inputs, baselines):
            return tuple(multiplier * 2 for multiplier in multipliers)

        attr = attr_method.attribute(
            inputs,
            baselines,
            custom_attribution_func=custom_attr_func,
            return_convergence_delta=True,
        )

        assertTensorAlmostEqual(self, attr[0], expected, 1e-19)


def _create_inps_and_base_for_deeplift_neuron_layer_testing() -> Tuple[
    Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]
]:
    x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

    b1 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    b2 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    inputs = (x1, x2)
    baselines = (b1, b2)

    return inputs, baselines


def _create_inps_and_base_for_deepliftshap_neuron_layer_testing() -> Tuple[
    Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]
]:
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
