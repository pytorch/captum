import base64
from collections import namedtuple
from io import BytesIO
from typing import Callable, List, Union

from captum.attr._utils import visualization as viz

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

FeatureOutput = namedtuple("FeatureOutput", "name base modified type contribution")


def _convert_img_base64(img):
    buff = BytesIO()

    plt.imsave(buff, img)
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


class Transformer(object):
    def __init__(self, transform: Callable[[Tensor], Tensor], name=None):
        self.transform = transform
        self.name = name


class BaseFeature(object):
    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Transformer, List[Transformer]],
        input_transforms: Union[Transformer, List[Transformer]],
        visualization_transform: Transformer,
    ):
        self.name = name
        self.baseline_transforms = baseline_transforms
        self.input_transforms = input_transforms
        self.visualization_transform = visualization_transform

    def visualization_type(self) -> str:
        raise NotImplementedError

    def visualize(self, attribution, data, label) -> FeatureOutput:
        raise NotImplementedError


class ImageFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        input_transforms: Union[Transformer, List[Transformer]],
        baseline_transforms: Union[Transformer, List[Transformer]],
    ):
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=None,
        )

    def visualization_type(self) -> str:
        return "image"

    def visualize(self, attribution, data, label) -> FeatureOutput:
        data_t = np.transpose(data.cpu().detach().numpy(), (1, 2, 0))
        attribution_t = np.transpose(
            attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )

        img_integrated_gradient_overlay = viz.visualize_image(
            attribution_t,
            data_t,
            clip_above_percentile=99,
            clip_below_percentile=0,
            overlay=True,
            mask_mode=True,
        )
        ig_64 = _convert_img_base64(img_integrated_gradient_overlay)
        img_64 = _convert_img_base64(data_t)

        return FeatureOutput(
            name=self.name,
            base=img_64,
            modified=ig_64,
            type=self.visualization_type(),
            contribution=100,  # TODO implement contribution
        )
