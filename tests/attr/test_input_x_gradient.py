#!/usr/bin/env python3

import torch

from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.noise_tunnel import NoiseTunnel

from .helpers.classification_models import SoftmaxModel
from .helpers.utils import assertArraysAlmostEqual, BaseTest
from .test_saliency import _get_basic_config, _get_multiargs_basic_config


class Test(BaseTest):
    def test_input_x_gradient_test_basic_vanilla(self):
        self._input_x_gradient_base_assert(*_get_basic_config())

    def test_input_x_gradient_test_basic_smoothgrad(self):
        self._input_x_gradient_base_assert(*_get_basic_config(), nt_type="smoothgrad")

    def test_input_x_gradient_test_basic_vargrad(self):
        self._input_x_gradient_base_assert(*_get_basic_config(), nt_type="vargrad")

    def test_saliency_test_basic_multi_variable_vanilla(self):
        self._input_x_gradient_base_assert(*_get_multiargs_basic_config())

    def test_saliency_test_basic_multi_variable_smoothgrad(self):
        self._input_x_gradient_base_assert(
            *_get_multiargs_basic_config(), nt_type="smoothgrad"
        )

    def test_saliency_test_basic_multi_vargrad(self):
        self._input_x_gradient_base_assert(
            *_get_multiargs_basic_config(), nt_type="vargrad"
        )

    def test_input_x_gradient_classification_vanilla(self):
        self._input_x_gradient_classification_assert()

    def test_input_x_gradient_classification_smoothgrad(self):
        self._input_x_gradient_classification_assert(nt_type="smoothgrad")

    def test_input_x_gradient_classification_vargrad(self):
        self._input_x_gradient_classification_assert(nt_type="vargrad")

    def _input_x_gradient_base_assert(
        self,
        model,
        inputs,
        expected_grads,
        additional_forward_args=None,
        nt_type="vanilla",
    ):
        input_x_grad = InputXGradient(model)
        if nt_type == "vanilla":
            attributions = input_x_grad.attribute(
                inputs, additional_forward_args=additional_forward_args
            )
        else:
            nt = NoiseTunnel(input_x_grad)
            attributions = nt.attribute(
                inputs,
                nt_type=nt_type,
                n_samples=10,
                stdevs=0.0002,
                additional_forward_args=additional_forward_args,
            )

        if isinstance(attributions, tuple):
            for input, attribution, expected_grad in zip(
                inputs, attributions, expected_grads
            ):
                if nt_type == "vanilla":
                    assertArraysAlmostEqual(
                        attribution.reshape(-1), (expected_grad * input).reshape(-1)
                    )
                self.assertEqual(input.shape, attribution.shape)
        else:
            if nt_type == "vanilla":
                assertArraysAlmostEqual(
                    attributions.reshape(-1),
                    (expected_grads * inputs).reshape(-1),
                    delta=0.5,
                )
            self.assertEqual(inputs.shape, attributions.shape)

    def _input_x_gradient_classification_assert(self, nt_type="vanilla"):
        num_in = 5
        input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        target = torch.tensor(5)

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        input_x_grad = InputXGradient(model.forward)
        if nt_type == "vanilla":
            attributions = input_x_grad.attribute(input, target)
            output = model(input)[:, target]
            output.backward()
            expercted = input.grad * input
            self.assertEqual(
                expercted.detach().numpy().tolist(),
                attributions.detach().numpy().tolist(),
            )
        else:
            nt = NoiseTunnel(input_x_grad)
            attributions = nt.attribute(
                input, nt_type=nt_type, n_samples=10, stdevs=1.0, target=target
            )

        self.assertAlmostEqual(attributions.shape, input.shape)
