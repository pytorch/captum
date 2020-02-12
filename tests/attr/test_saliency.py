#!/usr/bin/env python3
from typing import Any, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Module
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.saliency import Saliency
from captum.attr._utils.gradient import compute_gradients
from captum.attr._utils.typing import TensorOrTupleOfTensors

from .helpers.basic_models import BasicModel, BasicModel5_MultiArgs
from .helpers.classification_models import SoftmaxModel
from .helpers.utils import BaseTest, assertArraysAlmostEqual


def _get_basic_config() -> Tuple[Module, Tensor, Tensor, Any]:
    input = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, 7.0], requires_grad=True)
    # manually percomputed gradients
    grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
    return BasicModel(), input, grads, None


def _get_multiargs_basic_config() -> Tuple[
    Module, Tuple[Tensor, ...], Tuple[Tensor, ...], Any
]:
    model = BasicModel5_MultiArgs()
    additional_forward_args = ([2, 3], 1)
    inputs = (
        torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
        torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
    )
    grads = compute_gradients(
        model, inputs, additional_forward_args=additional_forward_args
    )
    return model, inputs, grads, additional_forward_args


class Test(BaseTest):
    def test_saliency_test_basic_vanilla(self) -> None:
        self._saliency_base_assert(*_get_basic_config())

    def test_saliency_test_basic_smoothgrad(self) -> None:
        self._saliency_base_assert(*_get_basic_config(), nt_type="smoothgrad")

    def test_saliency_test_basic_vargrad(self) -> None:
        self._saliency_base_assert(*_get_basic_config(), nt_type="vargrad")

    def test_saliency_test_basic_multi_variable_vanilla(self) -> None:
        self._saliency_base_assert(*_get_multiargs_basic_config())

    def test_saliency_test_basic_multi_variable_smoothgrad(self) -> None:
        self._saliency_base_assert(*_get_multiargs_basic_config(), nt_type="smoothgrad")

    def test_saliency_test_basic_multi_vargrad(self) -> None:
        self._saliency_base_assert(*_get_multiargs_basic_config(), nt_type="vargrad")

    def test_saliency_classification_vanilla(self) -> None:
        self._saliency_classification_assert()

    def test_saliency_classification_smoothgrad(self) -> None:
        self._saliency_classification_assert(nt_type="smoothgrad")

    def test_saliency_classification_vargrad(self) -> None:
        self._saliency_classification_assert(nt_type="vargrad")

    def _saliency_base_assert(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensors,
        expected: TensorOrTupleOfTensors,
        additional_forward_args: Any = None,
        nt_type: str = "vanilla",
    ) -> None:
        saliency = Saliency(model)
        if nt_type == "vanilla":
            attributions = saliency.attribute(
                inputs, additional_forward_args=additional_forward_args
            )
        else:
            nt = NoiseTunnel(saliency)
            attributions = nt.attribute(
                inputs,
                nt_type=nt_type,
                n_samples=10,
                stdevs=0.0000002,
                additional_forward_args=additional_forward_args,
            )

        for input, attribution, expected_attr in zip(inputs, attributions, expected):
            if nt_type == "vanilla":
                self._assert_attribution(attribution, expected_attr)
            self.assertEqual(input.shape, attribution.shape)

    def _assert_attribution(self, attribution: Tensor, expected: Tensor) -> None:
        expected = torch.abs(expected)
        assertArraysAlmostEqual(
            expected.detach().numpy().flatten().tolist(),
            attribution.detach().numpy().flatten().tolist(),
            delta=0.5,
        )

    def _saliency_classification_assert(self, nt_type: str = "vanilla") -> None:
        num_in = 5
        input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        target = torch.tensor(5)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        saliency = Saliency(model)

        if nt_type == "vanilla":
            attributions = saliency.attribute(input, target)

            output = model(input)[:, target]
            output.backward()
            expected = torch.abs(cast(Tensor, input.grad))
            self.assertEqual(
                expected.detach().numpy().tolist(),
                attributions.detach().numpy().tolist(),
            )
        else:
            nt = NoiseTunnel(saliency)
            attributions = nt.attribute(
                input, nt_type=nt_type, n_samples=10, stdevs=0.0002, target=target
            )
        self.assertEqual(input.shape, attributions.shape)
