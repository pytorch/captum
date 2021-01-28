#!/usr/bin/env python3
import unittest
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn

import captum.optim._core.loss as opt_loss
from captum.optim._core.output_hook import AbortForwardException, ModuleOutputsHook
from tests.helpers.basic import BaseTest, assertArraysAlmostEqual

CHANNEL_ACTIVATION_0_LOSS = 1
CHANNEL_ACTIVATION_1_LOSS = 1


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Conv2d(1, 2, 1)
        # Initialize weights and biases for
        # easy reproducibility
        self.layer.weight.data.fill_(0)
        self.layer.bias.data.fill_(1)

    def forward(self, x):
        return self.layer(x)


def get_loss_value(model, loss, input_shape=[1, 1, 1, 1]):
    if hasattr(loss.target, "__iter__"):
        hooks = ModuleOutputsHook(loss.target)
    else:
        hooks = ModuleOutputsHook([loss.target])
    with suppress(AbortForwardException):
        model(torch.zeros(*input_shape))
    module_outputs = hooks.consume_outputs()
    loss_value = loss(module_outputs)
    try:
        return loss_value.item()
    except ValueError:
        return loss_value.detach().numpy()


class TestChannelActivation(BaseTest):
    def test_channel_activation_0(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS)

    def test_channel_activation_1(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ChannelActivation(model.layer, 1)
        self.assertAlmostEqual(get_loss_value(model, loss), CHANNEL_ACTIVATION_1_LOSS)


class TestNeuronActivation(BaseTest):
    def test_neuron_activation_0(self) -> None:
        model = SimpleModel()
        loss = opt_loss.NeuronActivation(model.layer, 0)
        self.assertAlmostEqual(get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS)


class TestTotalVariation(BaseTest):
    def test_total_variation(self) -> None:
        model = SimpleModel()
        loss = opt_loss.TotalVariation(model.layer)
        self.assertAlmostEqual(get_loss_value(model, loss), 0.0)


class TestL1(BaseTest):
    def test_l1(self) -> None:
        model = SimpleModel()
        loss = opt_loss.L1(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS,
        )


class TestL2(BaseTest):
    def test_l2(self) -> None:
        model = SimpleModel()
        loss = opt_loss.L2(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            (CHANNEL_ACTIVATION_0_LOSS ** 2 + CHANNEL_ACTIVATION_1_LOSS ** 2) ** 0.5,
            places=5,
        )


class TestActivationInterpolation(BaseTest):
    def test_activation_interpolation_0_1(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ActivationInterpolation(
            target1=model.layer,
            channel_index1=0,
            target2=model.layer,
            channel_index2=1,
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss, input_shape=[2, 1, 1, 1]),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS,
        )


class TestAlignment(BaseTest):
    def test_alignment(self) -> None:
        model = SimpleModel()
        loss = opt_loss.Alignment(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss, input_shape=[2, 1, 1, 1]), 0.0
        )


class TestNeuronDirection(BaseTest):
    def test_neuron_direction(self) -> None:
        model = SimpleModel()
        loss = opt_loss.NeuronDirection(model.layer, vec=torch.ones(1, 1, 1, 1))
        a = 1
        b = [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        cos_sim = np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cos_sim = np.sum(cos_sim)
        self.assertAlmostEqual(get_loss_value(model, loss), cos_sim, places=6)


class TestTensorDirection(BaseTest):
    def test_tensor_direction(self) -> None:
        model = SimpleModel()
        loss = opt_loss.TensorDirection(model.layer, vec=torch.ones(1, 1, 1, 1))
        a = 1
        b = [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        cos_sim = np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cos_sim = np.sum(cos_sim)
        self.assertAlmostEqual(get_loss_value(model, loss), cos_sim, places=6)


class TestActivationWeights(BaseTest):
    def test_activation_weights_0(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ActivationWeights(model.layer, weights=torch.zeros(1))
        assertArraysAlmostEqual(get_loss_value(model, loss), np.zeros((1, 2, 1)))

    def test_activation_weights_1(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ActivationWeights(
            model.layer, weights=torch.ones(1), neuron=True
        )
        assertArraysAlmostEqual(
            get_loss_value(model, loss),
            [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS],
        )


class TestCompositeLoss(BaseTest):
    def test_negative(self) -> None:
        model = SimpleModel()
        loss = -opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(get_loss_value(model, loss), -CHANNEL_ACTIVATION_0_LOSS)

    def test_addition(self) -> None:
        model = SimpleModel()
        loss = (
            opt_loss.ChannelActivation(model.layer, 0)
            + opt_loss.ChannelActivation(model.layer, 1)
            + 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS + 1,
        )

    def test_subtraction(self) -> None:
        model = SimpleModel()
        loss = (
            opt_loss.ChannelActivation(model.layer, 0)
            - opt_loss.ChannelActivation(model.layer, 1)
            - 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            CHANNEL_ACTIVATION_0_LOSS - CHANNEL_ACTIVATION_1_LOSS - 1,
        )

    def test_multiplication(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ChannelActivation(model.layer, 0) * 10
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS * 10
        )

    def test_division(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ChannelActivation(model.layer, 0) / 10
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS / 10
        )

    def test_pow(self) -> None:
        model = SimpleModel()
        loss = opt_loss.ChannelActivation(model.layer, 0) ** 2
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS ** 2
        )


if __name__ == "__main__":
    unittest.main()
