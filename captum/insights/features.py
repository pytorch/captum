#!/usr/bin/env python3
import base64
import warnings
from collections import namedtuple
from io import BytesIO
from typing import Callable, List, Optional, Union

from captum.attr._utils import visualization as viz
from captum.attr._utils.common import safe_div

import numpy as np

FeatureOutput = namedtuple("FeatureOutput", "name base modified type contribution")


def _convert_figure_base64(fig):
    buff = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()  # removes padding
    fig.savefig(buff, format="png")
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


class BaseFeature:
    def __init__(
        self,
        name: str,
        baseline_transforms: Optional[Union[Callable, List[Callable]]],
        input_transforms: Optional[Union[Callable, List[Callable]]],
        visualization_transform: Optional[Callable],
    ):
        self.name = name
        self.baseline_transforms = baseline_transforms
        self.input_transforms = input_transforms
        self.visualization_transform = visualization_transform

    def visualization_type(self) -> str:
        raise NotImplementedError

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        raise NotImplementedError


class ImageFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Callable, List[Callable]],
        input_transforms: Union[Callable, List[Callable]],
        visualization_transform: Optional[Callable] = None,
    ):
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=visualization_transform,
        )

    def visualization_type(self) -> str:
        return "image"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        attribution = attribution.squeeze()
        data = data.squeeze()
        data_t = np.transpose(data.cpu().detach().numpy(), (1, 2, 0))
        attribution_t = np.transpose(
            attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )

        orig_fig, _ = viz.visualize_image_attr(
            attribution_t, data_t, method="original_image", use_pyplot=False
        )
        attr_fig, _ = viz.visualize_image_attr(
            attribution_t,
            data_t,
            method="heat_map",
            sign="absolute_value",
            use_pyplot=False,
        )

        img_64 = _convert_figure_base64(orig_fig)
        attr_img_64 = _convert_figure_base64(attr_fig)

        return FeatureOutput(
            name=self.name,
            base=img_64,
            modified=attr_img_64,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )


class TextFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Callable, List[Callable]],
        input_transforms: Union[Callable, List[Callable]],
        visualization_transform: Callable,
    ):
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=visualization_transform,
        )

    def visualization_type(self) -> str:
        return "text"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        text = self.visualization_transform(data)

        attribution = attribution.squeeze(0)
        data = data.squeeze(0)
        attribution = attribution.sum(dim=1)

        # L-Infinity norm
        attr_max = abs(attribution).max()
        normalized_attribution = safe_div(
            attribution, attr_max, default_value=attribution
        )

        modified = [x * 100 for x in normalized_attribution.tolist()]

        return FeatureOutput(
            name=self.name,
            base=text,
            modified=modified,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )


class GeneralFeature(BaseFeature):
    def __init__(self, name: str, categories: List[str]):
        super().__init__(
            name,
            baseline_transforms=None,
            input_transforms=None,
            visualization_transform=None,
        )
        self.categories = categories

    def visualization_type(self) -> str:
        return "general"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        attribution = attribution.squeeze(0)
        data = data.squeeze(0)

        # L-2 norm
        l2_norm = attribution.norm()
        normalized_attribution = safe_div(
            attribution, l2_norm, default_value=attribution
        )

        modified = [x * 100 for x in normalized_attribution.tolist()]

        base = [f"{c}: {d:.2f}" for c, d in zip(self.categories, data.tolist())]
        return FeatureOutput(
            name=self.name,
            base=base,
            modified=modified,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )
