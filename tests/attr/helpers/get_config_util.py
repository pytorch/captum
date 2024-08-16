# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Any, Tuple

import torch
from captum._utils.gradient import compute_gradients
from tests.helpers.basic_models import BasicModel, BasicModel5_MultiArgs
from torch import Tensor
from torch.nn import Module


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def get_basic_config() -> Tuple[Module, Tensor, Tensor, Any]:
    input = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, 7.0], requires_grad=True).T
    # manually percomputed gradients
    grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
    return BasicModel(), input, grads, None


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def get_multiargs_basic_config() -> (
    Tuple[Module, Tuple[Tensor, ...], Tuple[Tensor, ...], Any]
):
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


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def get_multiargs_basic_config_large() -> (
    Tuple[Module, Tuple[Tensor, ...], Tuple[Tensor, ...], Any]
):
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
