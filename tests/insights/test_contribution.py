from __future__ import print_function
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from captum.insights.api import AttributionVisualizer, Data
from captum.insights.features import ImageFeature, BaseFeature, FeatureOutput

from tests.attr.helpers.utils import BaseTest

class RealFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Callable, List[Callable]],
        input_transforms: Union[Callable, List[Callable]],
        visualization_transforms: Optional[Union[Callable, List[Callable]]] = None,
    ):
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
    def __init__(self, feature_extraction=False):
        super().__init__()
        self.feature_extraction = feature_extraction

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        if not self.feature_extraction:
            self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16)

        if not self.feature_extraction:
            x = self.fc(x)
        return x


class TinyMultiModal(nn.Module):
    def __init__(self, input_size=256, pretrained=False):
        super().__init__()
        if pretrained:
            self.img_model = _get_cnn(feature_extraction=True)
        else:
            self.img_model = TinyCnn(feature_extraction=True)

        self.misc_model = nn.Sequential(nn.Linear(input_size, 16), nn.ReLU())
        self.fc = nn.Linear(16 * 2, 10)

    def forward(self, img, misc):
        img = self.img_model(img)
        misc = self.misc_model(misc)
        x = torch.cat((img, misc), dim=-1)
        return self.fc(x)


def _labelled_img_data(num_samples=10, width=16, height=16, depth=3, num_labels=10):
    for i in range(num_samples):
        yield torch.randn(depth, height, width), torch.randint(num_labels, (1,))


<<<<<<< HEAD
def _multi_modal_data(img_dataset, feature_size=256):
    def misc_data(length, feature_size=256):
=======
def multi_modal_data(img_dataset, feature_size=None):
    def misc_data(length, feature_size=None):
>>>>>>> allow to provide arguments to ig.attribute
        for i in range(length):
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
        yield Data(inputs=tuple(x), labels=y)


class Test(BaseTest):
    def test_one_feature(self):
        batch_size = 3
        classes = _get_classes()
        dataset = list(_labelled_img_data(num_labels=len(classes), num_samples=2*batch_size))

        # NOTE: using DataLoader to batch the inputs -- since it is required
        data_loader = torch.utils.data.DataLoader(
            list(dataset), batch_size=batch_size, shuffle=False, num_workers=2
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

        outputs = visualizer.visualize(n_steps=2)

        for output in outputs:
            contribs = torch.stack(
                [feature.feature_outputs.contribution for feature in output]
            )
            total_contrib = torch.sum(torch.abs(contribs))
            self.assertAlmostEqual(total_contrib.item(), 1.0, places=6)

    def test_multi_features(self):
        batch_size = 3
        classes = _get_classes()
        img_dataset = list(_labelled_img_data(num_labels=len(classes), num_samples=2*batch_size))

        misc_feature_size = 5
        dataset = _multi_modal_data(
            img_dataset=img_dataset, feature_size=misc_feature_size
        )
        # NOTE: using DataLoader to batch the inputs -- since it is required
        data_loader = torch.utils.data.DataLoader(
            list(dataset), batch_size=batch_size, shuffle=False, num_workers=2
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

        outputs = visualizer.visualize(n_steps=2)

        for output in outputs:
            contribs = torch.stack(
                [feature.feature_outputs.contribution for feature in output]
            )
            total_contrib = torch.sum(torch.abs(contribs))
            self.assertAlmostEqual(total_contrib.item(), 1.0, places=6)

    # TODO: add test for multiple models (related to TODO in captum/insights/api.py)
    #
    # TODO: add test to make the attribs == 0 -- error occurs
    #       I know (through manual testing) that this breaks some existing code
