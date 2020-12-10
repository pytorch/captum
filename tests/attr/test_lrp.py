#!/usr/bin/env python3

import torch
import torch.nn as nn

from captum.attr import LRP, InputXGradient
from captum.attr._utils.lrp_rules import (
    Alpha1_Beta0_Rule,
    EpsilonRule,
    GammaRule,
    IdentityRule,
)

from ..helpers.basic import BaseTest, assertTensorAlmostEqual
from ..helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    SimpleLRPModel,
)


def _get_basic_config():
    input = torch.arange(16).view(1, 1, 4, 4).float()
    return BasicModel_ConvNet_One_Conv(), input


def _get_rule_config():
    relevance = torch.tensor([[[-0.0, 3.0]]])
    layer = nn.modules.Conv1d(1, 1, 2, bias=False)
    nn.init.constant_(layer.weight.data, 2)
    activations = torch.tensor([[[1.0, 5.0, 7.0]]])
    input = torch.tensor([[2, 0, -2]])
    return relevance, layer, activations, input


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
        assertTensorAlmostEqual(
            self,
            relevance,
            torch.Tensor(
                [[[[0, 1, 2, 3], [0, 5, 6, 7], [0, 9, 10, 11], [0, 0, 0, 0]]]]
            ),
        )

    def test_lrp_simple_attributions(self):
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([18.0, 36.0, 54.0]))

    def test_lrp_simple_attributions_batch(self):
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()
        model.linear2.rule = EpsilonRule()
        lrp = LRP(model)
        inputs = torch.cat((inputs, 3 * inputs))
        relevance, delta = lrp.attribute(
            inputs, target=0, return_convergence_delta=True
        )
        self.assertEqual(relevance.shape, inputs.shape)
        self.assertEqual(delta.shape[0], inputs.shape[0])
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[18.0, 36.0, 54.0], [54.0, 108.0, 162.0]])
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
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        _ = model(inputs)
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[0.0269, 0.0537, 0.0806]])
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
            self, relevance.data, torch.tensor([[28 / 3, 104 / 3, 52]])
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
        assertTensorAlmostEqual(self, relevance, torch.tensor([[12, 33.6, 50.4]]))

    def test_lrp_Identity(self):
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2
        model.eval()
        model.linear.rule = IdentityRule()
        model.linear2.rule = EpsilonRule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([24.0, 36.0, 36.0]))

    def test_lrp_simple2_attributions(self):
        model, input = _get_simple_model2()
        lrp = LRP(model)
        relevance = lrp.attribute(input, 0)
        self.assertEqual(relevance.shape, input.shape)

    def test_lrp_skip_connection(self):
        # A custom addition module needs to be used so that relevance is
        # propagated correctly.
        class Addition_Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x1, x2):
                return x1 + x2

        class SkipConnection(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(5)
                self.add = Addition_Module()

            def forward(self, input):
                x = self.add(self.linear(input), input)
                return x

        model = SkipConnection()
        input = torch.Tensor([[2, 3]])
        model.add.rule = EpsilonRule()
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)
        assertTensorAlmostEqual(self, relevance, torch.Tensor([[10, 18]]))

    def test_lrp_maxpool1D(self):
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
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
        assertTensorAlmostEqual(self, relevance, torch.Tensor([[[0.0, 0.0], [10, 12]]]))

    def test_lrp_maxpool2D(self):
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maxpool = nn.MaxPool2d(2)

            def forward(self, input):
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[1.0, 2.0], [5.0, 6.0]]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input)
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[[[0.0, 0.0], [0.0, 6.0]]]])
        )

    def test_lrp_maxpool3D(self):
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maxpool = nn.MaxPool3d(2)

            def forward(self, input):
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input)
        assertTensorAlmostEqual(
            self,
            relevance,
            torch.Tensor([[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 8.0]]]]]),
        )

    def test_lrp_multi(self):
        model = BasicModel_MultiLayer()
        input = torch.Tensor([[1, 2, 3]])
        add_input = 0
        output = model(input)
        output_add = model(input, add_input=add_input)
        self.assertTrue(torch.equal(output, output_add))
        lrp = LRP(model)
        attributions = lrp.attribute(input, target=0)
        attributions_add_input = lrp.attribute(
            input, target=0, additional_forward_args=(add_input,)
        )
        self.assertTrue(torch.equal(attributions, attributions_add_input))

    def test_lrp_multi_inputs(self):
        model = BasicModel_MultiLayer()
        input = torch.Tensor([[1, 2, 3]])
        input = (input, 3 * input)
        lrp = LRP(model)
        attributions, delta = lrp.attribute(
            input, target=0, return_convergence_delta=True
        )
        self.assertEqual(len(input), len(delta))
        assertTensorAlmostEqual(self, attributions[0], torch.Tensor([[16, 32, 48]]))

    def test_lrp_ixg_equivalency(self):
        model, inputs = _get_simple_model()
        lrp = LRP(model)
        attributions_lrp = lrp.attribute(inputs)
        ixg = InputXGradient(model)
        attributions_ixg = ixg.attribute(inputs)
        assertTensorAlmostEqual(
            self, attributions_lrp, attributions_ixg
        )  # Divide by score because LRP relevance is normalized.
