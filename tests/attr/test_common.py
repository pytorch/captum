#!/usr/bin/env python3

import torch
from captum.attr._core.noise_tunnel import SUPPORTED_NOISE_TUNNEL_TYPES
from captum.attr._utils.common import _validate_input, _validate_noise_tunnel_type
from tests.helpers.basic import BaseTest


class Test(BaseTest):
    def test_validate_input(self) -> None:
        with self.assertRaises(AssertionError):
            _validate_input((torch.tensor([-1.0, 1.0]),), (torch.tensor([-2.0]),))
            _validate_input(
                (torch.tensor([-1.0, 1.0]),), (torch.tensor([-1.0, 1.0]),), n_steps=-1
            )
            _validate_input(
                (torch.tensor([-1.0, 1.0]),),
                (torch.tensor([-1.0, 1.0]),),
                method="abcde",
            )
        _validate_input((torch.tensor([-1.0]),), (torch.tensor([-2.0]),))
        _validate_input(
            (torch.tensor([-1.0]),), (torch.tensor([-2.0]),), method="gausslegendre"
        )

    def test_validate_nt_type(self) -> None:
        with self.assertRaises(AssertionError):
            _validate_noise_tunnel_type("abc", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad_sq", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("vargrad", SUPPORTED_NOISE_TUNNEL_TYPES)
