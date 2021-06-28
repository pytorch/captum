#!/usr/bin/env python3
from typing import Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from captum.attr import LRP, InputXGradient
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import (
    AlphaBetaRule,
    EpsilonRule,
    FlatRule,
    GammaRule,
    IdentityRule,
    WSquaredRule,
    ZBoundRule,
    ZPlusRule,
)
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    SimpleLRPModel,
)


def _get_basic_config() -> Tuple[Module, Tensor]:
    input = torch.arange(16).view(1, 1, 4, 4).float()
    return BasicModel_ConvNet_One_Conv(), input


def _get_rule_config() -> Tuple[Tensor, Module, Tensor, Tensor]:
    relevance = torch.tensor([[[-0.0, 3.0]]])
    layer = nn.modules.Conv1d(1, 1, 2, bias=False)
    nn.init.constant_(layer.weight.data, 2)
    activations = torch.tensor([[[1.0, 5.0, 7.0]]])
    input = torch.tensor([[2, 0, -2]])
    return relevance, layer, activations, input


def _get_simple_model(inplace: bool = False) -> Tuple[Module, Tensor]:
    model = SimpleLRPModel(inplace)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])

    return model, inputs


def _get_simple_model2(inplace: bool = False) -> Tuple[Module, Tensor]:
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


def _get_two_layer_model() -> torch.nn.Module:
    """returns a simple model comprised of two layers."""

    class TwoLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer_1 = torch.nn.Linear(in_features=2, out_features=2, bias=False)
            self.layer_2 = torch.nn.Linear(in_features=2, out_features=2, bias=False)

            self.layer_1.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [1, -2]]))
            self.layer_2.weight = torch.nn.Parameter(torch.Tensor([[1, -2], [1, 3]]))

        def forward(self, x):
            out = self.layer_1(x)
            out = self.layer_2(out)
            return out

    model = TwoLayer()
    model.eval()

    return model


def _get_non_sequential() -> torch.nn.Module:
    class NonSequential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = torch.nn.Linear(in_features=2, out_features=2, bias=False)
            self.linear_2 = torch.nn.Linear(in_features=2, out_features=2, bias=False)

            self.linear_1.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [1, -2]]))
            self.linear_2.weight = torch.nn.Parameter(torch.Tensor([[1, -2], [1, 3]]))
            self.add = Addition_Module()

        def forward(self, input_):
            add_1 = self.linear_1(input_)
            add_2 = self.linear_2(input_)
            return self.add(add_1, add_2)

    model = NonSequential()
    model.eval()

    return model


class Test(BaseTest):
    def test_lrp_creator(self) -> None:
        model, _ = _get_basic_config()
        model.conv1.rule = 1  # type: ignore
        self.assertRaises(TypeError, LRP, model)

    def test_lrp_creator_activation(self) -> None:
        model, inputs = _get_basic_config()
        model.add_module("sigmoid", nn.Sigmoid())
        lrp = LRP(model)
        self.assertRaises(TypeError, lrp.attribute, inputs)

    def test_lrp_basic_attributions(self) -> None:
        model, inputs = _get_basic_config()
        logits = model(inputs)
        _, classIndex = torch.max(logits, 1)
        lrp = LRP(model)
        relevance, delta = lrp.attribute(
            inputs, cast(int, classIndex.item()), return_convergence_delta=True
        )
        self.assertEqual(delta.item(), 0)  # type: ignore
        self.assertEqual(relevance.shape, inputs.shape)  # type: ignore
        assertTensorAlmostEqual(
            self,
            relevance,
            torch.Tensor(
                [[[[0, 1, 2, 3], [0, 5, 6, 7], [0, 9, 10, 11], [0, 0, 0, 0]]]]
            ),
        )

    def test_lrp_simple_attributions(self) -> None:
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()  # type: ignore
        model.linear2.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([18.0, 36.0, 54.0]))

    def test_lrp_simple_attributions_batch(self) -> None:
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()  # type: ignore
        model.linear2.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        inputs = torch.cat((inputs, 3 * inputs))
        relevance, delta = lrp.attribute(
            inputs, target=0, return_convergence_delta=True
        )
        self.assertEqual(relevance.shape, inputs.shape)  # type: ignore
        self.assertEqual(delta.shape[0], inputs.shape[0])  # type: ignore
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[18.0, 36.0, 54.0], [54.0, 108.0, 162.0]])
        )

    def test_lrp_simple_repeat_attributions(self) -> None:
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = GammaRule()
        model.linear2.rule = ZPlusRule()
        output = model(inputs)
        lrp = LRP(model)
        _ = lrp.attribute(inputs)
        output_after = model(inputs)
        assertTensorAlmostEqual(self, output, output_after)

    def test_lrp_simple_inplaceReLU(self) -> None:
        model_default, inputs = _get_simple_model()
        model_inplace, _ = _get_simple_model(inplace=True)
        for model in [model_default, model_inplace]:
            model.eval()
            model.linear.rule = EpsilonRule()  # type: ignore
            model.linear2.rule = EpsilonRule()  # type: ignore
        lrp_default = LRP(model_default)
        lrp_inplace = LRP(model_inplace)
        relevance_default = lrp_default.attribute(inputs)
        relevance_inplace = lrp_inplace.attribute(inputs)
        assertTensorAlmostEqual(self, relevance_default, relevance_inplace)

    def test_lrp_simple_tanh(self) -> None:
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

    def test_lrp_simple_attributions_GammaRule(self) -> None:
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2  # type: ignore
        model.eval()
        model.linear.rule = GammaRule(gamma=1)  # type: ignore
        model.linear2.rule = GammaRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(
            self, relevance.data, torch.tensor([[28 / 3, 104 / 3, 52]])  # type: ignore
        )

    def test_lrp_simple_attributions_AlphaBeta(self) -> None:
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2  # type: ignore
        model.eval()
        model.linear.rule = ZPlusRule()
        model.linear2.rule = ZPlusRule()
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([[12, 33.6, 50.4]]))

    def test_lrp_Identity(self) -> None:
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2  # type: ignore
        model.eval()
        model.linear.rule = IdentityRule()  # type: ignore
        model.linear2.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)
        assertTensorAlmostEqual(self, relevance, torch.tensor([24.0, 36.0, 36.0]))

    def test_lrp_simple2_attributions(self) -> None:
        model, input = _get_simple_model2()
        lrp = LRP(model)
        relevance = lrp.attribute(input, 0)
        self.assertEqual(relevance.shape, input.shape)  # type: ignore

    def test_lrp_skip_connection(self) -> None:
        # A custom addition module needs to be used so that relevance is
        # propagated correctly.
        class Addition_Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
                return x1 + x2

        class SkipConnection(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(5)
                self.add = Addition_Module()

            def forward(self, input: Tensor) -> Module:
                x = self.add(self.linear(input), input)
                return x

        model = SkipConnection()
        input = torch.Tensor([[2, 3]])
        model.add.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)
        assertTensorAlmostEqual(self, relevance, torch.Tensor([[10, 18]]))

    def test_lrp_maxpool1D(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(2.0)
                self.maxpool = nn.MaxPool1d(2)

            def forward(self, input: Tensor) -> Module:
                return self.maxpool(self.linear(input))

        model = MaxPoolModel()
        input = torch.tensor([[[1.0, 2.0], [5.0, 6.0]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)
        assertTensorAlmostEqual(self, relevance, torch.Tensor([[[0.0, 0.0], [10, 12]]]))

    def test_lrp_maxpool2D(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maxpool = nn.MaxPool2d(2)

            def forward(self, input: Tensor) -> Module:
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[1.0, 2.0], [5.0, 6.0]]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input)
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[[[0.0, 0.0], [0.0, 6.0]]]])
        )

    def test_lrp_maxpool3D(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maxpool = nn.MaxPool3d(2)

            def forward(self, input: Tensor) -> Module:
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

    def test_lrp_multi(self) -> None:
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
        self.assertTrue(
            torch.equal(attributions, attributions_add_input)  # type: ignore
        )  # type: ignore

    def test_lrp_multi_inputs(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.Tensor([[1, 2, 3]])
        input = (input, 3 * input)
        lrp = LRP(model)
        attributions, delta = lrp.attribute(
            input, target=0, return_convergence_delta=True
        )
        self.assertEqual(len(input), 2)
        assertTensorAlmostEqual(self, attributions[0], torch.Tensor([[16, 32, 48]]))
        assertTensorAlmostEqual(self, delta, torch.Tensor(0))

    def test_lrp_ixg_equivalency(self) -> None:
        model, inputs = _get_simple_model()
        lrp = LRP(model)
        attributions_lrp = lrp.attribute(inputs)
        ixg = InputXGradient(model)
        attributions_ixg = ixg.attribute(inputs)
        assertTensorAlmostEqual(self, attributions_lrp, attributions_ixg)

    def test_epsilon_rule(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = EpsilonRule()
        model.layer_2.rule = EpsilonRule()
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor([[4 / 18, 14 / 18]])

        assertTensorAlmostEqual(self, attr, oracle)

    def test_gamma_rule(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        gamma_1 = 1
        gamma_2 = 1
        model.layer_1.rule = GammaRule(gamma=gamma_1)
        model.layer_2.rule = GammaRule(gamma=gamma_2)
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor(
            [
                [
                    ((1 + gamma_1) * (60 + 18 * gamma_1))
                    / (18 * (3 + gamma_1) * (5 + gamma_1)),
                    (210 + 66 * gamma_1) / (18 * (3 + gamma_1) * (5 + gamma_1)),
                ]
            ]
        )

        assertTensorAlmostEqual(self, attr, oracle)

    def test_identity_rule(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = IdentityRule()
        model.layer_2.rule = EpsilonRule()
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor([[3 / 18, 15 / 18]])

        assertTensorAlmostEqual(self, attr, oracle)

    def test_w_squared_rule(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = WSquaredRule()
        model.layer_2.rule = WSquaredRule()
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor([[23 / 100, 77 / 100]])

        assertTensorAlmostEqual(self, attr, oracle)

    def test_alpha_beta_rule_alpha1(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = AlphaBetaRule(alpha=1)
        model.layer_2.rule = AlphaBetaRule(alpha=1)
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor([[4 / 18, 14 / 18]])

        assertTensorAlmostEqual(self, attr, oracle)

    def test_alpha_beta_rule_alpha2(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        alpha_1 = 2
        alpha_2 = 2
        model.layer_1.rule = AlphaBetaRule(alpha=alpha_1)
        model.layer_2.rule = AlphaBetaRule(alpha=alpha_2)
        attr = lrp.attribute(input_, target=0)

        oracle = -7 * torch.Tensor(
            [
                [
                    (alpha_1 * (2 * alpha_2 + 3)) / 15,
                    (alpha_1 * (-2 * alpha_2 + 12)) / 15,
                ]
            ]
        )

        assertTensorAlmostEqual(self, attr, oracle)

    def test_z_bound_rule(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = ZBoundRule(-2, 2)
        model.layer_2.rule = ZBoundRule(-2, 2)
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor([[(303) / (7 * 11 * 13), (2 * 349) / (7 * 11 * 13)]])

        assertTensorAlmostEqual(self, attr, oracle)

    def test_flat_rule(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = FlatRule()
        model.layer_2.rule = FlatRule()
        attr = lrp.attribute(input_, target=1)

        oracle = 18 * torch.Tensor([[1 / 2, 1 / 2]])

        assertTensorAlmostEqual(self, attr, oracle)

    def test_alpha1_eq_zplus_for_pos_input(self) -> None:
        model = _get_two_layer_model()
        input_ = torch.Tensor([[1, 0]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.layer_1.rule = AlphaBetaRule(alpha=1)
        model.layer_2.rule = AlphaBetaRule(alpha=1)
        attr_1 = lrp.attribute(input_, target=1).detach()

        model.layer_1.rule = ZPlusRule()
        model.layer_2.rule = ZPlusRule()
        attr_2 = lrp.attribute(input_, target=1).detach()

        assertTensorAlmostEqual(self, attr_1, attr_2)

    def test_alpha_beta_non_sequential(self) -> None:
        model = _get_non_sequential()
        input_ = torch.Tensor([[1, -2]])
        input_.requires_grad = True

        lrp = LRP(model)
        model.add.rule = EpsilonRule()
        model.linear_1.rule = AlphaBetaRule(alpha=1)
        model.linear_2.rule = AlphaBetaRule(alpha=1)
        attr = lrp.attribute(input_, target=0)

        oracle = torch.Tensor([[2, 6]])

        assertTensorAlmostEqual(self, attr, oracle)
