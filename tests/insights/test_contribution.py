#!/usr/bin/env python3

import unittest
from typing import Callable, List, Union

import torch
import torch.nn as nn
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.app import FilterConfig
from captum.insights.attr_vis.features import BaseFeature, FeatureOutput, ImageFeature
from tests.helpers.basic import BaseTest


class RealFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Callable, List[Callable]],
        input_transforms: Union[Callable, List[Callable]],
        visualization_transforms: Union[None, Callable, List[Callable]] = None,
    ) -> None:
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=None,
        )

    def visualization_type(self):
        return "real"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        return FeatureOutput(
            name=self.name,
            base=data,
            modified=data,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )


def _get_classes():
    classes = [
        "Plane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    return classes


class TinyCnn(nn.Module):
    def __init__(self, feature_extraction=False) -> None:
        super().__init__()
        self.feature_extraction = feature_extraction

        self.conv1 = nn.Conv2d(3, 3, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        if not self.feature_extraction:
            self.conv2 = nn.Conv2d(3, 10, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))

        if not self.feature_extraction:
            x = self.conv2(x)
            x = x.view(-1, 10)
        else:
            x = x.view(-1, 12)

        return x


class TinyMultiModal(nn.Module):
    def __init__(self, input_size=256, pretrained=False) -> None:
        super().__init__()
        if pretrained:
            self.img_model = _get_cnn(feature_extraction=True)
        else:
            self.img_model = TinyCnn(feature_extraction=True)

        self.misc_model = nn.Sequential(nn.Linear(input_size, 12), nn.ReLU())
        self.fc = nn.Linear(12 * 2, 10)

    def forward(self, img, misc):
        img = self.img_model(img)
        misc = self.misc_model(misc)
        x = torch.cat((img, misc), dim=-1)
        return self.fc(x)


def _labelled_img_data(num_samples=10, width=8, height=8, depth=3, num_labels=10):
    for _ in range(num_samples):
        yield torch.empty(depth, height, width).uniform_(0, 1), torch.randint(
            num_labels, (1,)
        )


def _multi_modal_data(img_dataset, feature_size=256):
    def misc_data(length, feature_size=None):
        for _ in range(length):
            yield torch.randn(feature_size)

    misc_dataset = misc_data(length=len(img_dataset), feature_size=feature_size)

    # re-arrange dataset
    for (img, label), misc in zip(img_dataset, misc_dataset):
        yield ((img, misc), label)


def _get_cnn(feature_extraction=False):
    return TinyCnn(feature_extraction=feature_extraction)


def _get_multimodal(input_size=256):
    return TinyMultiModal(input_size=input_size, pretrained=True)


def to_iter(data_loader):
    # TODO: not sure how to make this cleaner
    for x, y in data_loader:
        # if it's not multi input
        # NOTE: torch.utils.data.DataLoader returns a list in this case
        if not isinstance(x, list):
            x = (x,)
        yield Batch(inputs=tuple(x), labels=y)


class Test(BaseTest):
    def test_one_feature(self):
        batch_size = 2
        classes = _get_classes()
        dataset = list(
            _labelled_img_data(num_labels=len(classes), num_samples=batch_size)
        )

        # NOTE: using DataLoader to batch the inputs
        # since AttributionVisualizer requires the input to be of size `B x ...`
        data_loader = torch.utils.data.DataLoader(
            list(dataset), batch_size=batch_size, shuffle=False, num_workers=0
        )

        visualizer = AttributionVisualizer(
            models=[_get_cnn()],
            classes=classes,
            features=[
                ImageFeature(
                    "Photo",
                    input_transforms=[lambda x: x],
                    baseline_transforms=[lambda x: x * 0],
                )
            ],
            dataset=to_iter(data_loader),
            score_func=None,
        )
        visualizer._config = FilterConfig(attribution_arguments={"n_steps": 2})

        outputs = visualizer.visualize()

        for output in outputs:
            total_contrib = sum(abs(f.contribution) for f in output[0].feature_outputs)
            self.assertAlmostEqual(total_contrib, 1.0, places=6)

    def test_multi_features(self):
        batch_size = 2
        classes = _get_classes()
        img_dataset = list(
            _labelled_img_data(num_labels=len(classes), num_samples=batch_size)
        )

        misc_feature_size = 2
        dataset = _multi_modal_data(
            img_dataset=img_dataset, feature_size=misc_feature_size
        )
        # NOTE: using DataLoader to batch the inputs since
        # AttributionVisualizer requires the input to be of size `N x ...`
        data_loader = torch.utils.data.DataLoader(
            list(dataset), batch_size=batch_size, shuffle=False, num_workers=0
        )

        visualizer = AttributionVisualizer(
            models=[_get_multimodal(input_size=misc_feature_size)],
            classes=classes,
            features=[
                ImageFeature(
                    "Photo",
                    input_transforms=[lambda x: x],
                    baseline_transforms=[lambda x: x * 0],
                ),
                RealFeature(
                    "Random",
                    input_transforms=[lambda x: x],
                    baseline_transforms=[lambda x: x * 0],
                ),
            ],
            dataset=to_iter(data_loader),
            score_func=None,
        )
        visualizer._config = FilterConfig(attribution_arguments={"n_steps": 2})

        outputs = visualizer.visualize()

        for output in outputs:
            total_contrib = sum(abs(f.contribution) for f in output[0].feature_outputs)
            self.assertAlmostEqual(total_contrib, 1.0, places=6)

    # TODO: add test for multiple models (related to TODO in captum/insights/api.py)
    #
    # TODO: add test to make the attribs == 0 -- error occurs
    #       I know (through manual testing) that this breaks some existing code


if __name__ == "__main__":
    unittest.main()
