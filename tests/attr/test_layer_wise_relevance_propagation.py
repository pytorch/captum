#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18
import torchvision.transforms as transforms

from captum.attr import visualization as viz
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.layer_wise_relevance_propagation import LRP
from captum.attr._utils.lrp_rules import (
    EpsilonRule,
    GammaRule,
    Alpha1_Beta0_Rule,
    suggested_rules
)

from .helpers.basic_models import BasicModel_ConvNet_One_Conv
from .helpers.utils import BaseTest, assertTensorAlmostEqual

from PIL import Image
import numpy as np
import os


def _get_basic_config():
    input = torch.arange(16).view(1, 1, 4, 4).float()
    grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
    expected_activations = torch.tensor([[54.1300, 54.1880, 54.2934, 53.9288]])
    return BasicModel_ConvNet_One_Conv(), input, grads, expected_activations, None


def _get_rule_config():
    relevance = torch.tensor([[[-0.0, 3.0]]])
    layer = nn.modules.Conv1d(1, 1, 2, bias=False)
    nn.init.constant_(layer.weight.data, 2)
    activations = torch.tensor([[[1.0, 5.0, 7.0]]])
    input = torch.tensor([2, 0, -2])
    return relevance, layer, activations, input


def _get_vgg_config():
    model = vgg16(pretrained=True)
    model.eval()
    image = _convert_image("images\\castle.jpg")
    image2 = _convert_image("images\\dackel.jpg")
    return model, image, image2


def _get_simple_model():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(3, 3, bias=False)
            self.linear.weight.data.fill_(2.0)
            self.relu = torch.nn.ReLU()
            self.linear2 = nn.Linear(3, 1, bias=False)
            self.linear2.weight.data.fill_(3.0)

        def forward(self, x):
            return self.linear2(self.relu(self.linear(x)))

    model = Model()
    return model


def _get_simple_model2():
    class MyModel(nn.Module):
        def __init__(self, inplace=False):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.lin.weight = nn.Parameter(torch.ones(2, 2))

        def forward(self, input):
            return self.lin(input)[0].unsqueeze(0)

    input = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    model = MyModel()

    return model, input


class Test(BaseTest):
    def test_lrp_creator(self):
        self._creator_error_assert(*_get_basic_config(), error="ruleType")

    def test_lrp_creator_activation(self):
        self._creator_error_assert(*_get_basic_config(), error="activationType")

    def test_lrp_basic_attributions(self):
        self._basic_attributions(*_get_basic_config())

    def test_lrp_simple_attributions(self):
        self._simple_attributions(_get_simple_model())

    def test_lrp_simple2_attributions(self):
        self._simple2_attributions(*_get_simple_model2())

    def test_lrp_vgg(self):
        self._attributions_assert_vgg(*_get_vgg_config())

    def _creator_error_assert(
        self, model, inputs, grads, expected_activations, expected, error
    ):
        if error == "ruleType":
            model.conv1.rule = 1
            self.assertRaises(TypeError, LRP, model)
        elif error == "activationType":
            model.add_module("sigmoid", nn.Sigmoid())
            lrp = LRP(model)
            self.assertRaises(TypeError, lrp.attribute, inputs)

    def _simple_attributions(self, model):
        model.eval()
        inputs = torch.tensor([1.0, 2.0, 3.0])
        output = model(inputs)
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance, torch.tensor([18 / 108, 36 / 108, 54 / 108])
        )

    def _simple2_attributions(self, model, input):
        output = model(input)
        lrp = LRP(model)
        relevance = lrp.attribute(input, 0)
        self.assertEqual(relevance.shape, input.shape)

    def _basic_attributions(self, model, inputs, grads, expected_activations, expected):
        logits = model(inputs)
        score, classIndex = torch.max(logits, 1)
        lrp = LRP(model)
        relevance = lrp.attribute(inputs, classIndex.item())
        self.assertEqual(relevance.shape, inputs.shape)

    def _attributions_assert_vgg(self, model, image, image2):

        classification = model(image)
        score, classIndex = torch.max(classification, 1)
        print(f"classindex: {classIndex}. score: {score}")
        rules = suggested_rules("vgg16")
        lrp = LRP(model)
        itg = InputXGradient(model)
        relevance = lrp.attribute(
            image, classIndex.item(), verbose=True
        )
        relevance_itg = itg.attribute(image, classIndex.item())
        itg_max = torch.max(relevance_itg)
        itg_min = torch.min(relevance_itg)
        new_max = torch.max(relevance)
        new_min = torch.min(relevance)
        print(f"min: {new_min}, max: {new_max}")
        print(f"input times gradient:\nmin: {itg_min}, max: {itg_max}")
        _ = viz.visualize_image_attr(
            np.transpose(relevance.data[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(image.data[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
            sign="all",
            method="heat_map",
            cmap="bwr",
            outlier_perc=2,
        )


def _convert_image(path):
    filepath = os.path.join(os.path.dirname(__file__), path)
    image = Image.open(filepath)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    loader = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image

