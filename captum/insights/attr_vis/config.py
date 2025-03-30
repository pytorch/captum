#!/usr/bin/env python3

# pyre-strict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

from captum.attr._core import (
    deep_lift,
    feature_ablation,
    guided_backprop_deconvnet,
    input_x_gradient,
    integrated_gradients,
    occlusion,
    saliency,
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


class StrConfig(NamedTuple):
    value: str
    type: str = "string"


Config = Union[NumberConfig, StrEnumConfig, StrConfig]

SUPPORTED_ATTRIBUTION_METHODS = [
    guided_backprop_deconvnet.Deconvolution,
    deep_lift.DeepLift,
    guided_backprop_deconvnet.GuidedBackprop,
    input_x_gradient.InputXGradient,
    integrated_gradients.IntegratedGradients,
    saliency.Saliency,
    feature_ablation.FeatureAblation,
    occlusion.Occlusion,
]


# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
class ConfigParameters(NamedTuple):
    params: Dict[str, Config]
    help_info: Optional[str] = None  # TODO fill out help for each method
    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    post_process: Optional[Dict[str, Callable[[Any], Any]]] = None


ATTRIBUTION_NAMES_TO_METHODS: Dict[
    str,
    Type[
        Union[
            deep_lift.DeepLift,
            feature_ablation.FeatureAblation,
            guided_backprop_deconvnet.Deconvolution,
            guided_backprop_deconvnet.GuidedBackprop,
            input_x_gradient.InputXGradient,
            integrated_gradients.IntegratedGradients,
            saliency.Saliency,
        ]
    ],
] = {
    # mypy bug - treating it as a type instead of a class
    cls.get_name(): cls  # type: ignore
    for cls in SUPPORTED_ATTRIBUTION_METHODS
}


def _str_to_tuple(s: Tuple[int, ...]) -> Tuple[int, ...]:
    if isinstance(s, tuple):
        return s
    return tuple([int(i) for i in s.split()])


ATTRIBUTION_METHOD_CONFIG: Dict[str, ConfigParameters] = {
    integrated_gradients.IntegratedGradients.get_name(): ConfigParameters(
        params={
            "n_steps": NumberConfig(value=25, limit=(2, None)),
            "method": StrEnumConfig(limit=SUPPORTED_METHODS, value="gausslegendre"),
        },
        post_process={"n_steps": int},
    ),
    feature_ablation.FeatureAblation.get_name(): ConfigParameters(
        params={"perturbations_per_eval": NumberConfig(value=1, limit=(1, 100))},
    ),
    occlusion.Occlusion.get_name(): ConfigParameters(
        params={
            "sliding_window_shapes": StrConfig(value=""),
            "strides": StrConfig(value=""),
            "perturbations_per_eval": NumberConfig(value=1, limit=(1, 100)),
        },
        post_process={
            "sliding_window_shapes": _str_to_tuple,
            "strides": _str_to_tuple,
            "perturbations_per_eval": int,
        },
    ),
}
