#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn

from captum.attr import LRP, InputXGradient
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule, GammaRule

from .helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    SimpleLRPModel,
)
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
        self.assertRaises(TypeError, LRP, model)

    def test_lrp_creator_activation(self):
        model, inputs = _get_basic_config()
        model.add_module("sigmoid", nn.Sigmoid())
        lrp = LRP(model)
        self.assertRaises(TypeError, lrp.attribute, inputs)

    def test_lrp_basic_attributions(self):
        model, inputs = _get_basic_config()
        logits = model(inputs)
        _, classIndex = torch.max(logits, 1)
        lrp = LRP(model)
        relevance, delta = lrp.attribute(
            inputs, classIndex.item(), return_convergence_delta=True
        )
        self.assertEqual(delta.item(), 0)
        self.assertEqual(relevance.shape, inputs.shape)

    def test_lrp_simple_attributions(self):
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance, torch.tensor([18 / 108, 36 / 108, 54 / 108])
        )

    def test_lrp_simple_repeat_attributions(self):
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = GammaRule()
        model.linear2.rule = Alpha1_Beta0_Rule()
        output = model(inputs)
        lrp = LRP(model)
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
        lrp_default = LRP(model_default)
        lrp_inplace = LRP(model_inplace)
        relevance_default = lrp_default.attribute(inputs)
        relevance_inplace = lrp_inplace.attribute(inputs)
        assertTensorAlmostEqual(self, relevance_default, relevance_inplace)

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
        _ = model(inputs)
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        # assertTensorAlmostEqual(self, relevance, torch.Tensor([[0.1186, 0.2372, 0.3558]])) # Result if gradient is used for propagation over tanh
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[1 / 6, 1 / 3, 1 / 2]])
        )  # Result if tanh is skipped for propagation

    def test_lrp_simple_attributions_GammaRule(self):
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2
        model.eval()
        model.linear.rule = GammaRule(gamma=1)
        model.linear2.rule = GammaRule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance.data, torch.tensor([21 / 216, 78 / 216, 117 / 216])
        )

    def test_lrp_simple_attributions_AlphaBeta(self):
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2
        model.eval()
        model.linear.rule = Alpha1_Beta0_Rule()
        model.linear2.rule = Alpha1_Beta0_Rule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([0.1250, 0.3500, 0.5250]))

    def test_lrp_simple2_attributions(self):
        model, input = _get_simple_model2()
        lrp = LRP(model)
        relevance = lrp.attribute(input, 0)
        self.assertEqual(relevance.shape, input.shape)

    def test_lrp_relu_hook(self):
        class OnlyRelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        model = OnlyRelu()
        input = torch.Tensor([-5])
        lrp = LRP(model)
        relevance = lrp.attribute(input)
        assertTensorAlmostEqual(self, relevance, torch.tensor([-5]))

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
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)
        denormalized_relevance = relevance * output[0, 1]
        # assertTensorAlmostEqual(self, comp_relevance, torch.Tensor([[10, 18]]))

    def test_lrp_maxpool1D(self):
        class MaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(2.0)
                self.maxpool = nn.MaxPool1d(2)

            def forward(self, input):
                return self.maxpool(self.linear(input))

        model = MaxPoolModel()
        input = torch.tensor([[[1.0, 2.0], [5.0, 6.0]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[[0.0, 0.0], [5 / 11, 6 / 11]]])
        )

    def test_lrp_maxpool2D(self):
        class MaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = nn.MaxPool2d(2)

            def forward(self, input):
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[1.0, 2.0], [5.0, 6.0]]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input)
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[[[0.0, 0.0], [0.0, 1.0]]]])
        )

    def test_lrp_maxpool3D(self):
        class MaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = nn.MaxPool3d(2)

            def forward(self, input):
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]]])
        output = model(input)
        lrp = LRP(model)
        relevance = lrp.attribute(input)
        assertTensorAlmostEqual(
            self,
            relevance,
            torch.Tensor([[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]]]),
        )

    def test_lrp_multi(self):
        model = BasicModel_MultiLayer()
        input = torch.Tensor([1, 2, 3])
        input.requires_grad = True

        add_input = 0
        output = model(input)
        output_add = model(input, add_input=add_input)
        self.assertTrue(torch.equal(output, output_add))
        lrp = LRP(model)
        attributions = lrp.attribute(input)
        attributions_add_input = lrp.attribute(
            input, additional_forward_args=(add_input,)
        )
        # due to problem with grad() function the results do not match (https://github.com/pytorch/pytorch/issues/35802)
        # self.assertTrue(torch.equal(attributions, attributions_add_input))

    def test_lrp_ixg_equivalency(self):
        model, inputs = _get_simple_model()
        lrp = LRP(model)
        attributions_lrp = lrp.attribute(inputs)
        ixg = InputXGradient(model)
        attributions_ixg = ixg.attribute(inputs)
        assertTensorAlmostEqual(
            self, attributions_lrp, attributions_ixg / 108
        )  # Divide by score because LRP relevance is normalized.
