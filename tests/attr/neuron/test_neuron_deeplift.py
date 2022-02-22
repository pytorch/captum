#!/usr/bin/env python3

from __future__ import print_function

import copy
from typing import Tuple, Union

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from tests.attr.layer.test_layer_deeplift import (
    _create_inps_and_base_for_deeplift_neuron_layer_testing,
    _create_inps_and_base_for_deepliftshap_neuron_layer_testing,
)
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_ConvNet_MaxPool3d,
    LinearMaxPoolLinearModel,
    ReLULinearModel,
)
from torch import Tensor


class Test(BaseTest):
    def test_relu_neuron_deeplift(self) -> None:
        model = ReLULinearModel(inplace=True)

        x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

        inputs = (x1, x2)

        neuron_dl = NeuronDeepLift(model, model.relu)
        attributions = neuron_dl.attribute(inputs, 0, attribute_to_neuron_input=False)
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 0.0, 0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[0.0, 0.0, 0.0]])

    def test_deeplift_compare_with_and_without_inplace(self) -> None:
        model1 = ReLULinearModel(inplace=True)
        model2 = ReLULinearModel()
        x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)
        inputs = (x1, x2)
        neuron_dl1 = NeuronDeepLift(model1, model1.relu)
        attributions1 = neuron_dl1.attribute(inputs, 0, attribute_to_neuron_input=False)

        neuron_dl2 = NeuronDeepLift(model2, model2.relu)
        attributions2 = neuron_dl2.attribute(inputs, 0, attribute_to_neuron_input=False)

        assertTensorAlmostEqual(self, attributions1[0], attributions2[0])
        assertTensorAlmostEqual(self, attributions1[1], attributions2[1])

    def test_linear_neuron_deeplift(self) -> None:
        model = ReLULinearModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        neuron_dl = NeuronDeepLift(model, model.l3)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=True
        )
        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[0.0, 0.0, 0.0]])

        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )
        self.assertTrue(neuron_dl.multiplies_by_inputs)
        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[6.0, 9.0, 0.0]])

    def test_linear_neuron_deeplift_wo_inp_marginal_effects(self) -> None:
        model = ReLULinearModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()

        neuron_dl = NeuronDeepLift(model, model.l3, multiply_by_inputs=False)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )
        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[2.0, 3.0, 0.0]])

    def test_relu_deeplift_with_custom_attr_func(self) -> None:
        model = ReLULinearModel()
        inputs, baselines = _create_inps_and_base_for_deeplift_neuron_layer_testing()
        neuron_dl = NeuronDeepLift(model, model.l3)
        expected = ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
        self._relu_custom_attr_func_assert(neuron_dl, inputs, baselines, expected)

    def test_relu_neuron_deeplift_shap(self) -> None:
        model = ReLULinearModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()

        neuron_dl = NeuronDeepLiftShap(model, model.relu)

        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )
        assertTensorAlmostEqual(self, attributions[0], [[0.0, 0.0, 0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[0.0, 0.0, 0.0]])

    def test_linear_neuron_deeplift_shap(self) -> None:
        model = ReLULinearModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()

        neuron_dl = NeuronDeepLiftShap(model, model.l3)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=True
        )
        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[0.0, 0.0, 0.0]])

        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )

        self.assertTrue(neuron_dl.multiplies_by_inputs)
        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[6.0, 9.0, 0.0]])

    def test_linear_neuron_deeplift_shap_wo_inp_marginal_effects(self) -> None:
        model = ReLULinearModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()

        neuron_dl = NeuronDeepLiftShap(model, model.l3, multiply_by_inputs=False)
        attributions = neuron_dl.attribute(
            inputs, 0, baselines, attribute_to_neuron_input=False
        )

        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[2.0, 3.0, 0.0]])

        attributions = neuron_dl.attribute(
            inputs, lambda x: x[:, 0], baselines, attribute_to_neuron_input=False
        )

        assertTensorAlmostEqual(self, attributions[0], [[-0.0, 0.0, -0.0]])
        assertTensorAlmostEqual(self, attributions[1], [[2.0, 3.0, 0.0]])

    def test_relu_deepliftshap_with_custom_attr_func(self) -> None:
        model = ReLULinearModel()
        (
            inputs,
            baselines,
        ) = _create_inps_and_base_for_deepliftshap_neuron_layer_testing()
        neuron_dl = NeuronDeepLiftShap(model, model.l3)
        expected = (torch.zeros(1, 3), torch.zeros(1, 3))
        self._relu_custom_attr_func_assert(neuron_dl, inputs, baselines, expected)

    def _relu_custom_attr_func_assert(
        self,
        attr_method: Union[NeuronDeepLift, NeuronDeepLiftShap],
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines,
        expected,
    ) -> None:
        def custom_attr_func(
            multipliers: Tuple[Tensor, ...],
            inputs: Tuple[Tensor, ...],
            baselines: Union[None, Tuple[Union[Tensor, int, float], ...]] = None,
        ) -> Tuple[Tensor, ...]:
            return tuple(multiplier * 0.0 for multiplier in multipliers)

        attr = attr_method.attribute(
            inputs, 0, baselines, custom_attribution_func=custom_attr_func
        )
        assertTensorAlmostEqual(self, attr[0], expected[0], 0.0)
        assertTensorAlmostEqual(self, attr[1], expected[1], 0.0)

    def test_lin_maxpool_lin_classification(self) -> None:
        inputs = torch.ones(2, 4)
        baselines = torch.tensor([[1, 2, 3, 9], [4, 8, 6, 7]]).float()

        model = LinearMaxPoolLinearModel()
        model_copy = copy.deepcopy(model)
        ndl = NeuronDeepLift(model, model.pool1)
        attr = ndl.attribute(inputs, neuron_selector=(0), baselines=baselines)

        ndl2 = NeuronDeepLift(model_copy, model_copy.lin2)
        attr2 = ndl2.attribute(
            inputs,
            neuron_selector=(0),
            baselines=baselines,
            attribute_to_neuron_input=True,
        )
        assertTensorAlmostEqual(self, attr, attr2)

    def test_convnet_maxpool2d_classification(self) -> None:
        inputs = 100 * torch.randn(2, 1, 10, 10)
        model = BasicModel_ConvNet()
        model_copy = copy.deepcopy(model)

        ndl = NeuronDeepLift(model, model.pool1)
        attr = ndl.attribute(inputs, neuron_selector=(0, 0, 0))

        ndl2 = NeuronDeepLift(model_copy, model_copy.conv2)
        attr2 = ndl2.attribute(
            inputs, neuron_selector=(0, 0, 0), attribute_to_neuron_input=True
        )

        assertTensorAlmostEqual(self, attr.sum(), attr2.sum())

    def test_convnet_maxpool3d_classification(self) -> None:
        inputs = 100 * torch.randn(2, 1, 10, 10, 10)
        model = BasicModel_ConvNet_MaxPool3d()
        model_copy = copy.deepcopy(model)

        ndl = NeuronDeepLift(model, model.pool1)
        attr = ndl.attribute(inputs, neuron_selector=(0, 0, 0, 0))

        ndl2 = NeuronDeepLift(model_copy, model_copy.conv2)
        attr2 = ndl2.attribute(
            inputs, neuron_selector=(0, 0, 0, 0), attribute_to_neuron_input=True
        )

        assertTensorAlmostEqual(self, attr.sum(), attr2.sum())
