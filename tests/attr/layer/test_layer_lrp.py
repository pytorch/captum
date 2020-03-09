#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn

from captum.attr._core.layer.layer_lrp import LayerLRP

from ..helpers.basic_models import BasicModel_ConvNet_One_Conv
from ..helpers.utils import BaseTest, assertTensorAlmostEqual


def _get_basic_config():
    input = torch.arange(16).view(1, 1, 4, 4).float()
    grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
    expected_activations = torch.tensor([[54.1300, 54.1880, 54.2934, 53.9288]])
    return BasicModel_ConvNet_One_Conv(), input, grads, expected_activations, None

#TODO: Check functionality and add testcases for layer LRP

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
    def test_lrp_basic_attributions(self):
        self._basic_attributions(*_get_basic_config())

    def test_lrp_simple_attributions(self):
        self._simple_attributions(_get_simple_model())

    def test_lrp_simple2_attributions(self):
        self._simple2_attributions(*_get_simple_model2())

    def _simple_attributions(self, model):
        model.eval()
        inputs = torch.tensor([1.0, 2.0, 3.0])
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance, torch.tensor([18 / 108, 36 / 108, 54 / 108])
        )

    def _simple2_attributions(self, model, input):
        lrp = LayerLRP(model, model.lin)
        relevance = lrp.attribute(input, 0)
        self.assertEqual(relevance.shape, input.shape)

    def _basic_attributions(self, model, inputs, grads, expected_activations, expected):
        logits = model(inputs)
        _, classIndex = torch.max(logits, 1)
        lrp = LayerLRP(model, model.lin)
        relevance = lrp.attribute(inputs, classIndex.item())
        self.assertEqual(relevance.shape, inputs.shape)
