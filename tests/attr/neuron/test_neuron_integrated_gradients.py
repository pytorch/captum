#!/usr/bin/env python3

import unittest
from typing import List, Optional, Tuple, Union, Any

import torch
from torch import Tensor
from torch.nn import Module

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)

from captum.attr._utils.typing import TensorOrTupleOfTensors

from ..helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from ..helpers.utils import (
    assertArraysAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)


class Test(BaseTest):
    def test_simple_ig_input_linear2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._ig_input_test_assert(net, net.linear2, inp, 0, [0.0, 390.0, 0.0])

    def test_simple_ig_input_linear1(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.linear1, inp, (0,), [0.0, 100.0, 0.0])

    def test_simple_ig_input_relu(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 6.0, 14.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.relu, inp, (0,), [0.0, 3.0, 7.0])

    def test_simple_ig_input_relu2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._ig_input_test_assert(net, net.relu, inp, 1, [0.0, 5.0, 4.0])

    def test_simple_ig_multi_input_linear2(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            (0,),
            ([[0.0, 156.0, 0.0]], [[0.0, 156.0, 0.0]], [[0.0, 78.0, 0.0]]),
            (4,),
        )

    def test_simple_ig_multi_input_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (0,),
            ([[0.0, 1.5, 3.5]], [[0.0, 1.5, 3.5]]),
            (inp3, 0.5),
        )

    def test_simple_ig_multi_input_relu_batch(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 80.0, 0.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 20.0, 0.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (0,),
            ([[0.0, 1.5, 3.5], [0.0, 40.0, 0.0]], [[0.0, 1.5, 3.5], [0.0, 10.0, 0.0]]),
            (inp3, 0.5),
        )

    def test_matching_output_gradient(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10, 10, requires_grad=True)
        self._ig_matching_test_assert(net, net.softmax, inp, baseline)

    def _ig_input_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensors,
        test_neuron: Union[int, Tuple[int, ...]],
        expected_input_ig: Union[List[float], Tuple[List[List[float]], ...]],
        additional_input: Any = None,
    ) -> None:
        for internal_batch_size in [None, 1, 20]:
            grad = NeuronIntegratedGradients(model, target_layer)
            attributions = grad.attribute(
                test_input,
                test_neuron,
                n_steps=200,
                method="gausslegendre",
                additional_forward_args=additional_input,
                internal_batch_size=internal_batch_size,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_input_ig, delta=0.1
            )

    def _ig_matching_test_assert(
        self,
        model: Module,
        output_layer: Module,
        test_input: Tensor,
        baseline: Optional[Tensor] = None,
    ) -> None:
        out = model(test_input)
        input_attrib = IntegratedGradients(model)
        ig_attrib = NeuronIntegratedGradients(model, output_layer)
        for i in range(out.shape[1]):
            ig_vals = input_attrib.attribute(test_input, target=i, baselines=baseline)
            neuron_ig_vals = ig_attrib.attribute(test_input, (i,), baselines=baseline)
            assertArraysAlmostEqual(
                ig_vals.reshape(-1).tolist(),
                neuron_ig_vals.reshape(-1).tolist(),
                delta=0.001,
            )
            self.assertEqual(neuron_ig_vals.shape, test_input.shape)


if __name__ == "__main__":
    unittest.main()
