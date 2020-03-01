#!/usr/bin/env python3
from typing import List, NamedTuple, Optional, Tuple

from captum.attr import (
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    Saliency,
)
from captum.attr._utils.approximation_methods import SUPPORTED_METHODS


class NumberConfig(NamedTuple):
    value: int = 1
    limit: Tuple[Optional[int], Optional[int]] = (None, None)
    type: str = "number"


class StrEnumConfig(NamedTuple):
    value: str
    limit: List[str]
    type: str = "enum"


SUPPORTED_ATTRIBUTION_METHODS = [
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    Saliency,
]

ATTRIBUTION_NAMES_TO_METHODS = {
    cls.get_name(): cls for cls in SUPPORTED_ATTRIBUTION_METHODS
}

ATTRIBUTION_METHOD_CONFIG = {
    IntegratedGradients.get_name(): {
        "n_steps": NumberConfig(value=25, limit=(2, None)),
        "method": StrEnumConfig(limit=SUPPORTED_METHODS, value="gausslegendre"),
    }
}
