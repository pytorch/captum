#!/usr/bin/env python3

import torch
import torch.nn as nn
from captum.attr import LayerLRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule, GammaRule

from ...helpers.basic import BaseTest, assertTensorAlmostEqual
from ...helpers.basic_models import BasicModel_ConvNet_One_Conv, SimpleLRPModel


def _get_basic_config():
    input = torch.arange(16).view(1, 1, 4, 4).float()
    return BasicModel_ConvNet_One_Conv(), input


def _get_simple_model(inplace=False):
    model = SimpleLRPModel(inplace)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])

    return model, inputs


def _get_simple_model2(inplace=False):
    class MyModel(nn.Module):
        def __init__(self, inplace) -> None:
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
        assertTensorAlmostEqual(
            self, relevance[0], torch.Tensor([[[0, 4], [31, 40]], [[0, 0], [-6, -15]]])
        )
        assertTensorAlmostEqual(self, delta, torch.Tensor([0]))

    def test_lrp_simple_attributions(self):
        model, inputs = _get_simple_model(inplace=False)
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp_upper = LayerLRP(model, model.linear2)
        relevance_upper, delta = lrp_upper.attribute(
            inputs, attribute_to_layer_input=True, return_convergence_delta=True
        )
        lrp_lower = LayerLRP(model, model.linear)
        relevance_lower = lrp_lower.attribute(inputs)
        assertTensorAlmostEqual(self, relevance_lower[0], relevance_upper[0])
        self.assertEqual(delta.item(), 0)

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

    def test_lrp_simple_tanh(self):
        class Model(nn.Module):
            def __init__(self) -> None:
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
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance[0], torch.Tensor([0.0537, 0.0537, 0.0537])
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
        assertTensorAlmostEqual(self, relevance[0], torch.tensor([24.0, 36.0, 36.0]))

    def test_lrp_simple_attributions_AlphaBeta(self):
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2
        model.eval()
        model.linear.rule = Alpha1_Beta0_Rule()
        model.linear2.rule = Alpha1_Beta0_Rule()
        lrp = LayerLRP(model, model.linear)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance[0], torch.tensor([24.0, 36.0, 36.0]))

    def test_lrp_simple_attributions_all_layers(self):
        model, inputs = _get_simple_model(inplace=False)
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        layers = [model.linear, model.linear2]
        lrp = LayerLRP(model, layers)
        relevance = lrp.attribute(inputs, attribute_to_layer_input=True)
        self.assertEqual(len(relevance), 2)
        assertTensorAlmostEqual(self, relevance[0][0], torch.tensor([18.0, 36.0, 54.0]))

    def test_lrp_simple_attributions_all_layers_delta(self):
        model, inputs = _get_simple_model(inplace=False)
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        layers = [model.linear, model.linear2]
        lrp = LayerLRP(model, layers)
        inputs = torch.cat((inputs, 2 * inputs))
        relevance, delta = lrp.attribute(
            inputs, attribute_to_layer_input=True, return_convergence_delta=True
        )
        self.assertEqual(len(relevance), len(delta))
        assertTensorAlmostEqual(
            self,
            relevance[0],
            torch.tensor([[18.0, 36.0, 54.0], [36.0, 72.0, 108.0]]),
        )
