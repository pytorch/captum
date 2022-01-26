#!/usr/bin/env python3
from typing import Any, Tuple, Union, cast

import torch
from captum._utils.gradient import compute_gradients
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.saliency import Saliency
from tests.helpers.basic import (
    BaseTest,
    assertTensorTuplesAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.helpers.basic_models import BasicModel, BasicModel5_MultiArgs
from tests.helpers.classification_models import SoftmaxModel
from torch import Tensor
from torch.nn import Module


def _get_basic_config() -> Tuple[Module, Tensor, Tensor, Any]:
    input = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, 7.0], requires_grad=True).T
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


def _get_multiargs_basic_config_large() -> Tuple[
    Module, Tuple[Tensor, ...], Tuple[Tensor, ...], Any
]:
    model = BasicModel5_MultiArgs()
    additional_forward_args = ([2, 3], 1)
    inputs = (
        torch.tensor(
            [[10.5, 12.0, 34.3], [43.4, 51.2, 32.0]], requires_grad=True
        ).repeat_interleave(3, dim=0),
        torch.tensor(
            [[1.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True
        ).repeat_interleave(3, dim=0),
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

    def test_saliency_test_basic_multivar_sg_n_samples_batch_size_2(self) -> None:
        attributions_batch_size = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="smoothgrad",
            n_samples_batch_size=2,
        )
        attributions = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="smoothgrad",
        )

        assertTensorTuplesAlmostEqual(self, attributions_batch_size, attributions)

    def test_saliency_test_basic_multivar_sg_n_samples_batch_size_3(self) -> None:
        attributions_batch_size = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="smoothgrad_sq",
            n_samples_batch_size=3,
        )
        attributions = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="smoothgrad_sq",
        )
        assertTensorTuplesAlmostEqual(self, attributions_batch_size, attributions)

    def test_saliency_test_basic_multivar_vg_n_samples_batch_size_1(self) -> None:
        attributions_batch_size = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="vargrad",
            n_samples_batch_size=1,
        )
        attributions = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="vargrad",
        )
        assertTensorTuplesAlmostEqual(self, attributions_batch_size, attributions)

    def test_saliency_test_basic_multivar_vg_n_samples_batch_size_6(self) -> None:
        attributions_batch_size = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="vargrad",
            n_samples_batch_size=6,
        )
        attributions = self._saliency_base_assert(
            *_get_multiargs_basic_config_large(),
            nt_type="vargrad",
        )
        assertTensorTuplesAlmostEqual(self, attributions_batch_size, attributions)

    def test_saliency_test_basic_multi_vargrad(self) -> None:
        self._saliency_base_assert(*_get_multiargs_basic_config(), nt_type="vargrad")

    def test_saliency_classification_vanilla(self) -> None:
        self._saliency_classification_assert()

    def test_saliency_classification_smoothgrad(self) -> None:
        self._saliency_classification_assert(nt_type="smoothgrad")

    def test_saliency_classification_vargrad(self) -> None:
        self._saliency_classification_assert(nt_type="vargrad")

    def test_saliency_grad_unchanged(self) -> None:
        model, inp, grads, add_args = _get_basic_config()
        inp.grad = torch.randn_like(inp)
        grad = inp.grad.detach().clone()
        self._saliency_base_assert(model, inp, grads, add_args)
        assertTensorTuplesAlmostEqual(self, inp.grad, grad, delta=0.0)

    def _saliency_base_assert(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Any = None,
        nt_type: str = "vanilla",
        n_samples_batch_size=None,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        saliency = Saliency(model)

        self.assertFalse(saliency.multiplies_by_inputs)

        if nt_type == "vanilla":
            attributions = saliency.attribute(
                inputs, additional_forward_args=additional_forward_args
            )
        else:
            nt = NoiseTunnel(saliency)
            attributions = nt.attribute(
                inputs,
                nt_type=nt_type,
                nt_samples=10,
                nt_samples_batch_size=n_samples_batch_size,
                stdevs=0.0000002,
                additional_forward_args=additional_forward_args,
            )

        for input, attribution, expected_attr in zip(inputs, attributions, expected):
            if nt_type == "vanilla":
                self._assert_attribution(attribution, expected_attr)
            self.assertEqual(input.shape, attribution.shape)

        return attributions

    def _assert_attribution(self, attribution: Tensor, expected: Tensor) -> None:
        expected = torch.abs(expected)
        if len(attribution.shape) == 0:
            assert (attribution - expected).abs() < 0.001
        else:
            assertTensorAlmostEqual(self, expected, attribution, delta=0.5, mode="max")

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
            assertTensorAlmostEqual(self, attributions, expected)
        else:
            nt = NoiseTunnel(saliency)
            attributions = nt.attribute(
                input, nt_type=nt_type, nt_samples=10, stdevs=0.0002, target=target
            )
        self.assertEqual(input.shape, attributions.shape)
