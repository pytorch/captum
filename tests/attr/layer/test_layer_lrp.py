#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn

from captum.attr._core.layer.layer_lrp import LayerLRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule, GammaRule

from ..helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    SimpleLRPModel,
)
from ..helpers.utils import BaseTest, assertTensorAlmostEqual


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


def _get_simple_model(inplace=False):
    model = SimpleLRPModel(inplace)
    inputs = torch.tensor([1.0, 2.0, 3.0])

    return model, inputs


def _get_simple_model2(inplace=False):
    class MyModel(nn.Module):
        def __init__(self, inplace):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.lin.weight = nn.Parameter(torch.ones(2, 2))
            self.relu = torch.nn.ReLU(inplace=inplace)

        def forward(self, input):
            return self.relu(self.lin(input))[0].unsqueeze(0)

    input = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    model = MyModel(inplace)

    return model, input


class Test(BaseTest):
    def test_lrp_creator(self):
        model, _ = _get_basic_config()
        model.conv1.rule = 1
        self.assertRaises(TypeError, LayerLRP, model, model.conv1)

    def test_lrp_creator_activation(self):
        model, inputs = _get_basic_config()
        model.add_module("sigmoid", nn.Sigmoid())
        lrp = LayerLRP(model, model.conv1)
        self.assertRaises(TypeError, lrp.attribute, inputs)

    def test_lrp_basic_attributions(self):
        model, inputs = _get_basic_config()
        logits = model(inputs)
        score, classIndex = torch.max(logits, 1)
        lrp = LayerLRP(model, model.conv1)
        relevance, delta = lrp.attribute(
            inputs, classIndex.item(), return_convergence_delta=True
        )
        self.assertEqual(delta.item(), 0)
        self.assertEqual(relevance.shape, torch.Size([1, 2, 2, 2]))

    def test_lrp_simple_attributions(self):
        model, inputs = _get_simple_model(inplace=False)
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        output = model(inputs)
        lrp_upper = LayerLRP(model, model.linear2)
        relevance_upper = lrp_upper.attribute(inputs, attribute_to_layer_input=True)
        lrp_lower = LayerLRP(model, model.linear)
        relevance_lower = lrp_lower.attribute(inputs)
        assertTensorAlmostEqual(self, relevance_lower, relevance_upper[0])

    def test_lrp_simple_repeat_attributions(self):
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = GammaRule()
        model.linear2.rule = Alpha1_Beta0_Rule()
        output = model(inputs)
        lrp = LayerLRP(model, model.linear)
        _ = lrp.attribute(inputs)
        output_after = model(inputs)
        assertTensorAlmostEqual(self, output, output_after)

    def test_lrp_simple_inplaceReLU(self):
        model_default, inputs = _get_simple_model()
        model_inplace, _ = _get_simple_model(inplace=True)
        for model in [model_default, model_inplace]:
            model.eval()
            model.linear.rule = EpsilonRule()
            model.linear2.rule = EpsilonRule()
        lrp_default = LayerLRP(model_default, model_default.linear2)
        lrp_inplace = LayerLRP(model_inplace, model_inplace.linear2)
        relevance_default = lrp_default.attribute(inputs, attribute_to_layer_input=True)
        relevance_inplace = lrp_inplace.attribute(inputs, attribute_to_layer_input=True)
        # assertTensorAlmostEqual(self, relevance_default, relevance_inplace)

    def test_lrp_simple_tanh(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(3, 3, bias=False)
                self.linear.weight.data.fill_(0.1)
                self.tanh = torch.nn.Tanh()
                self.linear2 = nn.Linear(3, 1, bias=False)
                self.linear2.weight.data.fill_(0.1)

            def forward(self, x):
                return self.linear2(self.tanh(self.linear(x)))

        model = Model()
        _, inputs = _get_simple_model()
        output = model(inputs)
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[1 / 3, 1 / 3, 1 / 3]])
        )  # Result if tanh is skipped for propagation

    def test_lrp_simple_attributions_GammaRule(self):
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2
        model.eval()
        model.linear.rule = GammaRule(gamma=1)
        model.linear2.rule = GammaRule()
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance.data, torch.tensor([0.2500, 0.3750, 0.3750])
        )

    def test_lrp_simple_attributions_AlphaBeta(self):
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2
        model.eval()
        model.linear.rule = Alpha1_Beta0_Rule()
        model.linear2.rule = Alpha1_Beta0_Rule()
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([0.2500, 0.3750, 0.3750]))

    def test_lrp_skip_connection(self):
        class SkipConnection(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(5)

            def forward(self, input):
                # TODO: Solve deviating behaviour between += input and +input
                x = self.linear(input) + input
                # x += input
                return x

        model = SkipConnection()
        input = torch.Tensor([[2, 3]])
        output = model(input)
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(input, target=1)
        denormalized_relevance = relevance * output[0, 1]
        # assertTensorAlmostEqual(self, comp_relevance, torch.Tensor([[10, 18]]))

    def test_lrp_simple_attributions_all_layers(self):
        model, inputs = _get_simple_model(inplace=False)
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp = LayerLRP(model, None)
        relevance = lrp.attribute(inputs, attribute_to_layer_input=True)
        self.assertEqual(len(relevance), 2)

    def test_lrp_simple_attributions_all_layers_delta(self):
        model, inputs = _get_simple_model(inplace=False)
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp = LayerLRP(model, None)
        inputs = torch.stack((inputs, 2 * inputs))
        relevance, delta = lrp.attribute(
            inputs, attribute_to_layer_input=True, return_convergence_delta=True
        )
        self.assertEqual(len(relevance), len(delta))
