from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from captum.insights.api import AttributionVisualizer, Data
from captum.insights.features import ImageFeature, RealFeature

from ..attr.helpers.utils import BaseTest


def get_classes():
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


class BasicCnn(nn.Module):
    def __init__(self, feature_extraction=False):
        super(BasicCnn, self).__init__()
        self.feature_extraction = feature_extraction

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(
            84, 10
        )  # note: not removing this in order to load the params
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        if not self.feature_extraction:
            x = self.fc3(x)
        return x


class BasicMultiModal(nn.Module):
    def __init__(self, input_size=256, pretrained=False):
        super(BasicMultiModal, self).__init__()
        if pretrained:
            self.img_model = get_pretrained_cnn(feature_extraction=True)
        else:
            self.img_model = BasicCnn(feature_extraction=True)
        self.misc_model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 84)
        )
        self.fc = nn.Linear(84 * 2, 10)

    def forward(self, img, misc):
        img = self.img_model(img)
        misc = self.misc_model(misc)
        x = torch.cat((img, misc), dim=-1)
        return self.fc(x)


def multi_modal_data(img_dataset, feature_size=256):
    def misc_data(length, feature_size=256):
        for i in range(length):
            yield torch.randn(feature_size)

    misc_dataset = misc_data(length=len(img_dataset), feature_size=feature_size)

    # re-arrange dataset
    for (img, label), misc in zip(img_dataset, misc_dataset):
        yield ((img, misc), label)


def get_pretrained_cnn(feature_extraction=False):
    net = BasicCnn(feature_extraction=feature_extraction)
    net.load_state_dict(torch.load("tutorials/models/cifar_torchvision.pt"))
    return net


def get_pretrained_multimodal(input_size=256):
    return BasicMultiModal(input_size=input_size, pretrained=True)


class Test(BaseTest):
    def test_one_feature(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

        def to_iter(data_loader):
            for x, y in data_loader:
                yield Data(inputs=(x,), labels=y)

        visualizer = AttributionVisualizer(
            models=[get_pretrained_cnn()],
            classes=get_classes(),
            features=[ImageFeature("Photo", 
                                   input_transforms=[lambda x: x], 
                                   baseline_transforms=[lambda x: x * 0])],
            dataset=to_iter(data_loader),
            score_func=None,
        )
        outputs = visualizer.visualize()

        for output in outputs:
            contribs = torch.stack(
                [feature.feature_outputs.contribution for feature in output]
            )
            total_contrib = torch.sum(torch.abs(contribs))
            self.assertAlmostEqual(total_contrib.item(), 1.0, places=6)

    def test_multi_features(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        img_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        misc_feature_size = 5
        dataset = multi_modal_data(
            img_dataset=img_dataset, feature_size=misc_feature_size
        )
        data_loader = torch.utils.data.DataLoader(list(dataset), batch_size=10, shuffle=False, num_workers=2)

        def input_transform(x):
            return x

        def to_iter(data_loader):
            for x, y in data_loader:
                yield Data(inputs=x, labels=y)

        visualizer = AttributionVisualizer(
            models=[
                get_pretrained_multimodal(input_size=misc_feature_size)
            ],  # some nn.Module
            classes=get_classes(),  # a list of classes, indices correspond to name
            features=[
                 ImageFeature("Photo", 
                              input_transforms=[input_transform],
                              baseline_transforms=[lambda x: x * 0]),
                RealFeature("Random", input_transforms=[lambda x: x], baseline_transforms=[lambda x: x * 0])
            ],
            dataset=to_iter(data_loader), 
            score_func=None,
        )

        outputs = visualizer.visualize()

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
