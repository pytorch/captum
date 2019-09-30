import base64
from collections import namedtuple
from io import BytesIO
from typing import Callable, List, Optional, Union

from captum.attr._utils import visualization as viz

import numpy as np

FeatureOutput = namedtuple("FeatureOutput", "name base modified type contribution")


def _convert_figure_base64(fig):
    buff = BytesIO()
    fig.savefig(buff, format="png", pad_inches=0.0)
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

    def visualize(self, attribution, data) -> FeatureOutput:
        raise NotImplementedError


class ImageFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        input_transforms: Union[Callable, List[Callable]],
        baseline_transforms: Union[Callable, List[Callable]],
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

    def visualize(self, attribution, data) -> FeatureOutput:
        attribution.squeeze_()
        data.squeeze_()
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
            contribution=100,  # TODO implement contribution
        )


class TextFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        input_transforms: Union[Callable, List[Callable]],
        baseline_transforms: Union[Callable, List[Callable]],
        visualization_transform: Optional[Callable],
    ):
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=visualization_transform,
        )

    def visualization_type(self) -> str:
        return "text"

    def visualize(self, attribution, data) -> FeatureOutput:
        text = self.visualization_transform(data)

        attribution.squeeze_(0)
        data.squeeze_(0)
        attribution = attribution.sum(dim=1)

        # L-Infinity norm
        normalized_attribution = attribution / abs(attribution).max()
        modified = [x * 100 for x in normalized_attribution.tolist()]

        return FeatureOutput(
            name=self.name,
            base=text,
            modified=modified,
            type=self.visualization_type(),
            contribution=100,  # TODO implement contribution
        )
