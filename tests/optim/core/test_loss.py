#!/usr/bin/env python3
import unittest
from typing import cast, List, Union

import captum.optim._core.loss as opt_loss
import numpy as np
import torch
from captum.optim.models import collect_activations
from packaging import version
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import BasicModel_ConvNet_Optim

CHANNEL_ACTIVATION_0_LOSS = 1.3
CHANNEL_ACTIVATION_1_LOSS = 1.3


def get_loss_value(
    model: torch.nn.Module, loss: opt_loss.Loss, input_shape: List[int] = [1, 3, 1, 1]
) -> Union[int, float, np.ndarray]:
    module_outputs = collect_activations(model, loss.target, torch.ones(*input_shape))
    loss_value = loss(module_outputs)
    try:
        return loss_value.item()
    except ValueError:
        return loss_value.detach()


class TestDeepDream(BaseTest):
    def test_channel_deepdream(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.DeepDream(model.layer)
        expected = torch.as_tensor(
            [[[CHANNEL_ACTIVATION_0_LOSS ** 2]], [[CHANNEL_ACTIVATION_1_LOSS ** 2]]]
        )[None, :]
        assertTensorAlmostEqual(self, get_loss_value(model, loss), expected, mode="max")


class TestChannelActivation(BaseTest):
    def test_channel_activation_0(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS, places=6
        )

    def test_channel_activation_1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 1)
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_1_LOSS, places=6
        )


class TestNeuronActivation(BaseTest):
    def test_neuron_activation_0(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.NeuronActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS, places=6
        )


class TestTotalVariation(BaseTest):
    def test_total_variation(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.TotalVariation(model.layer)
        self.assertAlmostEqual(get_loss_value(model, loss), 0.0)


class TestL1(BaseTest):
    def test_l1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.L1(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS,
            places=6,
        )


class TestL2(BaseTest):
    def test_l2(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.L2(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            (CHANNEL_ACTIVATION_0_LOSS ** 2 + CHANNEL_ACTIVATION_1_LOSS ** 2) ** 0.5,
            places=5,
        )


class TestDiversity(BaseTest):
    def test_diversity(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.Diversity(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss, input_shape=[2, 3, 1, 1]),
            -1,
        )


class TestActivationInterpolation(BaseTest):
    def test_activation_interpolation_0_1(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping Activation Interpolation test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationInterpolation(
            target1=model.layer,
            channel_index1=0,
            target2=model.layer,
            channel_index2=1,
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss, input_shape=[2, 3, 1, 1]),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS,
            places=6,
        )


class TestAlignment(BaseTest):
    def test_alignment(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.Alignment(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss, input_shape=[2, 3, 1, 1]), 0.0
        )


class TestNeuronDirection(BaseTest):
    def test_neuron_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.NeuronDirection(model.layer, vec=torch.ones(1, 1, 1, 1))
        a = 1
        b = [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        dot = np.sum(np.inner(a, b))
        self.assertAlmostEqual(get_loss_value(model, loss), dot, places=6)


class TestAngledNeuronDirection(BaseTest):
    def test_angled_neuron_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.AngledNeuronDirection(
            model.layer, vec=torch.ones(1, 2), cossim_pow=0
        )
        a = 1
        b = [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        dot = torch.sum(torch.as_tensor(np.inner(a, b))).item()
        output = torch.sum(cast(torch.Tensor, get_loss_value(model, loss)))
        self.assertAlmostEqual(output.item(), dot, places=6)

    def test_angled_neuron_direction_whitened(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.AngledNeuronDirection(
            model.layer,
            vec=torch.ones(1, 2),
            vec_whitened=torch.ones(2, 2),
            cossim_pow=0,
        )
        a = 1
        b = [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        dot = torch.sum(torch.as_tensor(np.inner(a, b))).item() * 2
        output = torch.sum(cast(torch.Tensor, get_loss_value(model, loss)))
        self.assertAlmostEqual(output.item(), dot, places=6)


class TestTensorDirection(BaseTest):
    def test_tensor_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.TensorDirection(model.layer, vec=torch.ones(1, 1, 1, 1))
        a = 1
        b = [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        dot = np.sum(np.inner(a, b))
        self.assertAlmostEqual(get_loss_value(model, loss), dot, places=6)


class TestActivationWeights(BaseTest):
    def test_activation_weights_0(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationWeights(model.layer, weights=torch.zeros(1))
        assertTensorAlmostEqual(
            self, get_loss_value(model, loss), torch.zeros(1, 2, 1, 1), mode="max"
        )

    def test_activation_weights_1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationWeights(
            model.layer, weights=torch.ones(1), neuron=True
        )
        assertTensorAlmostEqual(
            self,
            get_loss_value(model, loss),
            [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS],
            mode="max",
        )


class TestCompositeLoss(BaseTest):
    def test_negative(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = -opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss), -CHANNEL_ACTIVATION_0_LOSS, places=6
        )

    def test_addition(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = (
            opt_loss.ChannelActivation(model.layer, 0)
            + opt_loss.ChannelActivation(model.layer, 1)
            + 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS + 1,
            places=6,
        )

    def test_subtraction(self) -> None:
        model = BasicModel_ConvNet_Optim()
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
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) * 10
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS * 10, places=5
        )

    # def test_multiplication_error(self) -> None:
    #     model = BasicModel_ConvNet_Optim()
    #     with self.assertRaises(TypeError):
    #         opt_loss.ChannelActivation(model.layer, 0) * "string"
    #     with self.assertRaises(TypeError):
    #         opt_loss.ChannelActivation(model.layer, 0) * opt_loss.ChannelActivation(
    #             model.layer, 1
    #         )

    def test_division(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) / 10
        self.assertAlmostEqual(
            get_loss_value(model, loss), CHANNEL_ACTIVATION_0_LOSS / 10
        )

    # def test_division_error(self) -> None:
    #     model = BasicModel_ConvNet_Optim()
    #     with self.assertRaises(TypeError):
    #         opt_loss.ChannelActivation(model.layer, 0) / "string"
    #     with self.assertRaises(TypeError):
    #         opt_loss.ChannelActivation(model.layer, 0) / opt_loss.ChannelActivation(
    #             model.layer, 1
    #         )

    def test_pow(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) ** 2
        self.assertAlmostEqual(
            get_loss_value(model, loss),
            CHANNEL_ACTIVATION_0_LOSS ** 2,
            places=6,
        )

    # def test_pow_error(self) -> None:
    #     model = BasicModel_ConvNet_Optim()
    #     with self.assertRaises(TypeError):
    #         opt_loss.ChannelActivation(model.layer, 0) ** "string"
    #     with self.assertRaises(TypeError):
    #         opt_loss.ChannelActivation(model.layer, 0) ** opt_loss.ChannelActivation(
    #             model.layer, 1
    #         )

    def test_sum_loss_list(self) -> None:
        n_batch = 400
        model = torch.nn.Identity()
        loss_fn_list = [opt_loss.LayerActivation(model) for i in range(n_batch)]
        loss_fn = opt_loss.sum_loss_list(loss_fn_list)
        out = get_loss_value(model, loss_fn, [n_batch, 3, 1, 1])
        self.assertEqual(out, float(n_batch))

    def test_sum_loss_list_compose_add(self) -> None:
        n_batch = 400
        model = torch.nn.Identity()
        loss_fn_list = [opt_loss.LayerActivation(model) for i in range(n_batch)]
        loss_fn = opt_loss.sum_loss_list(loss_fn_list) + opt_loss.LayerActivation(model)
        out = get_loss_value(model, loss_fn, [n_batch, 3, 1, 1])
        self.assertEqual(out, float(n_batch + 1.0))
