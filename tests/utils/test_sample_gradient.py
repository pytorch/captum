#!/usr/bin/env python3

import unittest
from typing import Tuple, Callable

import torch
from captum._utils.sample_gradient import SampleGradientWrapper, SUPPORTED_MODULES
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_ConvNetWithPaddingDilation,
)
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_sample_grads_linear_sum(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        inp = (torch.randn(6, 3), torch.randn(6, 3))
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.sum(x), "sum")

    def test_sample_grads_linear_mean(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        inp = (20 * torch.randn(6, 3),)
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.mean(x))

    def test_sample_grads_conv_sum(self) -> None:
        model = BasicModel_ConvNet_One_Conv()
        inp = (123 * torch.randn(6, 1, 4, 4),)
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.sum(x), "sum")

    def test_sample_grads_conv_mean_multi_inp(self) -> None:
        model = BasicModel_ConvNet_One_Conv()
        inp = (20 * torch.randn(6, 1, 4, 4), 9 * torch.randn(6, 1, 4, 4))
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.mean(x))

    def test_sample_grads_modified_conv_mean(self) -> None:
        if torch.__version__ < "1.8":
            raise unittest.SkipTest(
                "Skipping sample gradient test with 3D linear module"
                "since torch version < 1.8"
            )

        model = BasicModel_ConvNetWithPaddingDilation()
        inp = (20 * torch.randn(6, 1, 5, 5),)
        self._compare_sample_grads_per_sample(
            model, inp, lambda x: torch.mean(x), "mean"
        )

    def test_sample_grads_modified_conv_sum(self) -> None:
        if torch.__version__ < "1.8":
            raise unittest.SkipTest(
                "Skipping sample gradient test with 3D linear module"
                "since torch version < 1.8"
            )

        model = BasicModel_ConvNetWithPaddingDilation()
        inp = (20 * torch.randn(6, 1, 5, 5),)
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.sum(x), "sum")

    def _compare_sample_grads_per_sample(
        self,
        model: Module,
        inputs: Tuple[Tensor, ...],
        loss_fn: Callable,
        loss_type: str = "mean",
    ):
        wrapper = SampleGradientWrapper(model)
        wrapper.add_hooks()
        out = model(*inputs)
        wrapper.compute_param_sample_gradients(loss_fn(out), loss_type)

        batch_size = inputs[0].shape[0]
        for i in range(batch_size):
            model.zero_grad()
            single_inp = tuple(inp[i : i + 1] for inp in inputs)
            out = model(*single_inp)
            loss_fn(out).backward()
            for layer in model.modules():
                if isinstance(layer, tuple(SUPPORTED_MODULES.keys())):
                    assertTensorAlmostEqual(
                        self,
                        layer.weight.grad,
                        layer.weight.sample_grad[i],  # type: ignore
                        mode="max",
                    )
                    assertTensorAlmostEqual(
                        self,
                        layer.bias.grad,
                        layer.bias.sample_grad[i],  # type: ignore
                        mode="max",
                    )
