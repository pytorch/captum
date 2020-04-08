#!/usr/bin/env python3

from typing import Any, Dict, List, Tuple, Type, cast

from torch.nn import Module

from captum.attr._utils.attribution import Attribution


def gen_test_name(
    prefix: str, test_name: str, algorithm: Type[Attribution], noise_tunnel: bool
) -> str:
    # Generates test name for dynamically generated tests
    return (
        prefix
        + "_"
        + test_name
        + "_"
        + algorithm.__name__
        + ("NoiseTunnel" if noise_tunnel else "")
    )


def parse_test_config(
    test_config: Dict,
) -> Tuple[List[Type[Attribution]], Module, Dict[str, Any], Module, bool, bool]:
    algorithms = cast(List[Type[Attribution]], test_config["algorithms"])
    model = test_config["model"]
    args = cast(Dict[str, Any], test_config["attribute_args"])
    layer = test_config["layer"] if "layer" in test_config else None
    noise_tunnel = (
        test_config["noise_tunnel"] if "noise_tunnel" in test_config else False
    )
    baseline_distr = (
        test_config["baseline_distr"] if "baseline_distr" in test_config else False
    )
    return algorithms, model, args, layer, noise_tunnel, baseline_distr
