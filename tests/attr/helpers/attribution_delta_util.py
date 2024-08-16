# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Tuple, Union

import torch
from captum._utils.typing import Tensor
from tests.helpers import BaseTest


def assert_attribution_delta(
    test: BaseTest,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    attributions: Union[Tensor, Tuple[Tensor, ...]],
    n_samples: int,
    delta: Tensor,
    delta_thresh: Union[float, Tensor] = 0.0006,
    is_layer: bool = False,
) -> None:
    if not is_layer:
        for input, attribution in zip(inputs, attributions):
            test.assertEqual(attribution.shape, input.shape)
    if isinstance(inputs, tuple):
        bsz = inputs[0].shape[0]
    else:
        bsz = inputs.shape[0]
    test.assertEqual([bsz * n_samples], list(delta.shape))

    delta = torch.mean(delta.reshape(bsz, -1), dim=1)
    assert_delta(test, delta, delta_thresh)


def assert_delta(
    test: BaseTest, delta: Tensor, delta_thresh: Union[Tensor, float] = 0.0006
) -> None:
    delta_condition = (delta.abs() < delta_thresh).all()
    test.assertTrue(
        delta_condition,
        "Sum of SHAP values {} does"
        " not match the difference of endpoints.".format(delta),
    )
