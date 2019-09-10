import base64
from collections import namedtuple
from io import BytesIO

from captum.attr._utils import visualization as viz

import numpy as np
from matplotlib import pyplot as plt

FeatureOutput = namedtuple("FeatureOutput", "name base modified type contribution")


def convert_img_base64(img, denormalize=False):
    if denormalize:
        img = img / 2 + 0.5

    buff = BytesIO()

    plt.imsave(buff, img)
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


class BaseFeature:
    def __init__(self, name: str):
        self.name = name

    def visualization_type(self):
        raise NotImplementedError


class ImageFeature(BaseFeature):
    def __init__(self, name: str):
        super().__init__(name)

    def visualization_type(self):
        return "image"

    def visualize(self, attribution, data, label, contribution):
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
        ig_64 = convert_img_base64(img_integrated_gradient_overlay)
        img_64 = convert_img_base64(data_t, True)

        return FeatureOutput(
            name=self.name,
            base=img_64,
            modified=ig_64,
            type=self.visualization_type(),
            contribution=contribution,
        )


class TextFeature(BaseFeature):
    def __init__(self, name: str):
        super().__init__(name)


class ComplexFeature(BaseFeature):
    def __init__(self, name: str):
        super().__init__(name)
