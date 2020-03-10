#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn

from captum.attr._core.lrp import LRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule, GammaRule

from .helpers.basic_models import BasicModel_ConvNet_One_Conv
from .helpers.utils import BaseTest, assertTensorAlmostEqual


def _get_basic_config():
    input = torch.arange(16).view(1, 1, 4, 4).float()
    return BasicModel_ConvNet_One_Conv(), input


def _get_rule_config():
    relevance = torch.tensor([[[-0.0, 3.0]]])
    layer = nn.modules.Conv1d(1, 1, 2, bias=False)
    nn.init.constant_(layer.weight.data, 2)
    activations = torch.tensor([[[1.0, 5.0, 7.0]]])
    input = torch.tensor([2, 0, -2])
    return relevance, layer, activations, input


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
            self.relu = torch.nn.ReLU()

        def forward(self, input):
            return self.relu(self.lin(input))[0].unsqueeze(0)

    input = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    model = MyModel()

    return model, input


class Test(BaseTest):
    def test_lrp_creator(self):
        model, _ = _get_basic_config()
        model.conv1.rule = 1
        self.assertRaises(TypeError, LRP, model)

    def test_lrp_creator_activation(self):
        model, inputs = _get_basic_config()
        model.add_module("sigmoid", nn.Sigmoid())
        lrp = LRP(model)
        self.assertRaises(TypeError, lrp.attribute, inputs)

    def test_lrp_basic_attributions(self):
        model, inputs = _get_basic_config()
        logits = model(inputs)
        score, classIndex = torch.max(logits, 1)
        lrp = LRP(model)
        relevance, delta = lrp.attribute(inputs, classIndex.item(), return_convergence_delta=True)
        self.assertEqual(delta.item(), 0)
        self.assertEqual(relevance.shape, inputs.shape)

    def test_lrp_simple_attributions(self):
        model = _get_simple_model()
        model.eval()
        inputs = torch.tensor([1.0, 2.0, 3.0])
        output = model(inputs)
        model.linear.rule = GammaRule(gamma=0)
        model.linear2.rule = Alpha1_Beta0_Rule(set_bias_to_zero=True)
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance, torch.tensor([18 / 108, 36 / 108, 54 / 108])
        )

    def test_lrp_simple2_attributions(self):
        model, input = _get_simple_model2()
        output = model(input)
        lrp = LRP(model)
        relevance = lrp.attribute(input, 0)
        self.assertEqual(relevance.shape, input.shape)

