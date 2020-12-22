#!/usr/bin/env python3
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import captum.optim._utils.models as model_utils
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestPadReflectiveA4D(BaseTest):
    def test_pad_reflective_a4d_b1_c1(self) -> None:
        b = 1
        c = 1
        x = torch.arange(0, b * c * 4 * 4).view(b, c, 4, 4).float()
        padding = [2] * 8

        x_out = model_utils.pad_reflective_a4d(x, padding)
        x_out_np = torch.as_tensor(np.pad(x.numpy(), (2), mode="reflect"))

        assertTensorAlmostEqual(self, x_out, x_out_np)

    def test_pad_reflective_a4d_b3_c3(self) -> None:
        b = 3
        c = 3
        x = torch.arange(0, b * c * 4 * 4).view(b, c, 4, 4).float()
        padding = [2] * 8

        x_out = model_utils.pad_reflective_a4d(x, padding)
        x_out_np = torch.as_tensor(np.pad(x.numpy(), (2), mode="reflect"))

        assertTensorAlmostEqual(self, x_out, x_out_np)

    def test_pad_reflective_a4d_zeros(self) -> None:
        b = 1
        c = 3
        x = torch.arange(0, b * c * 4 * 4).view(b, c, 4, 4).float()
        padding = [0] * 8

        x_out = model_utils.pad_reflective_a4d(x, padding)

        assertTensorAlmostEqual(self, x_out, x)


class TestConv2dSame(BaseTest):
    def test_conv2d_same(self) -> None:
        x = torch.ones(64, 32, 100, 20)

        expected_conv = torch.nn.Conv2d(32, 32, (4, 1))
        x_expected = expected_conv(F.pad(x, (0, 0, 2, 1)))

        conv2d_same = model_utils.Conv2dSame(32, 32, (3, 1))
        x_output = conv2d_same(x)

        self.assertEqual(tuple(x_output.size()), tuple(x_expected.size()))


class TestLocalResponseNormLayer(BaseTest):
    def test_local_response_norm_layer(self) -> None:
        size = 5
        alpha = 9.999999747378752e-05
        beta = 0.75
        k = 1

        x = torch.randn(32, 5, 24, 24)
        lrn_layer = model_utils.LocalResponseNormLayer(
            size=size, alpha=alpha, beta=beta, k=k
        )

        assertTensorAlmostEqual(
            self,
            lrn_layer(x),
            F.local_response_norm(x, size=size, alpha=alpha, beta=beta, k=k),
            0,
        )


class TestReluLayer(BaseTest):
    def test_relu_layer(self) -> None:
        x = torch.randn(1, 3, 4, 4)
        relu_layer = model_utils.ReluLayer()
        assertTensorAlmostEqual(self, relu_layer(x), F.relu(x), 0)


class TestRedirectedReluLayer(BaseTest):
    def test_forward_redirected_relu_layer(self) -> None:
        x = torch.randn(1, 3, 4, 4)
        layer = model_utils.RedirectedReluLayer()
        assertTensorAlmostEqual(self, layer(x), F.relu(x), 0)

    def test_backward_redirected_relu_layer(self) -> None:
        t_grad_input, t_grad_output = [], []

        def check_grad(self, grad_input, grad_output):
            t_grad_input.append(grad_input[0].clone().detach())
            t_grad_output.append(grad_output[0].clone().detach())

        rr_layer = model_utils.RedirectedReluLayer()
        x = torch.zeros(1, 3, 4, 4, requires_grad=True)
        rr_layer.register_backward_hook(check_grad)
        rr_loss = rr_layer(x * 1).mean()
        rr_loss.backward()

        assertTensorAlmostEqual(self, t_grad_input[0], t_grad_output[0], 0)


def check_is_not_instance(self, model, layer) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            check_is_not_instance(self, child, layer)


class TestReplaceLayers(BaseTest):
    def test_replace_layers(self) -> None:
        class BasicReluModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.relu(input)

        class BasicReluModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu1 = torch.nn.ReLU()
                self.relu2 = BasicReluModule()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.relu2(self.relu1(input))

        toy_model = BasicReluModel()
        old_layer = torch.nn.ReLU
        new_layer = model_utils.RedirectedReluLayer
        model_utils.replace_layers(toy_model, old_layer, new_layer)
        # Unittest can't run replace_layers correctly?
        model_utils.replace_layers(toy_model.relu2, old_layer, new_layer)

        check_is_not_instance(self, toy_model, old_layer)
        self.assertIsInstance(toy_model.relu1, new_layer)
        self.assertIsInstance(toy_model.relu2.relu, new_layer)


class TestGetLayers(BaseTest):
    def test_get_layers_pretrained_inceptionv1(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_layers test due to insufficient Torch version."
            )
        expected_list = [
            "conv1",
            "conv1_relu",
            "pool1",
            "localresponsenorm1",
            "conv2",
            "conv2_relu",
            "conv3",
            "conv3_relu",
            "localresponsenorm2",
            "pool2",
            "mixed3a",
            "mixed3a.conv_1x1",
            "mixed3a.conv_1x1_relu",
            "mixed3a.conv_3x3_reduce",
            "mixed3a.conv_3x3_reduce_relu",
            "mixed3a.conv_3x3",
            "mixed3a.conv_3x3_relu",
            "mixed3a.conv_5x5_reduce",
            "mixed3a.conv_5x5_reduce_relu",
            "mixed3a.conv_5x5",
            "mixed3a.conv_5x5_relu",
            "mixed3a.pool",
            "mixed3a.pool_proj",
            "mixed3a.pool_proj_relu",
            "mixed3b",
            "mixed3b.conv_1x1",
            "mixed3b.conv_1x1_relu",
            "mixed3b.conv_3x3_reduce",
            "mixed3b.conv_3x3_reduce_relu",
            "mixed3b.conv_3x3",
            "mixed3b.conv_3x3_relu",
            "mixed3b.conv_5x5_reduce",
            "mixed3b.conv_5x5_reduce_relu",
            "mixed3b.conv_5x5",
            "mixed3b.conv_5x5_relu",
            "mixed3b.pool",
            "mixed3b.pool_proj",
            "mixed3b.pool_proj_relu",
            "pool3",
            "mixed4a",
            "mixed4a.conv_1x1",
            "mixed4a.conv_1x1_relu",
            "mixed4a.conv_3x3_reduce",
            "mixed4a.conv_3x3_reduce_relu",
            "mixed4a.conv_3x3",
            "mixed4a.conv_3x3_relu",
            "mixed4a.conv_5x5_reduce",
            "mixed4a.conv_5x5_reduce_relu",
            "mixed4a.conv_5x5",
            "mixed4a.conv_5x5_relu",
            "mixed4a.pool",
            "mixed4a.pool_proj",
            "mixed4a.pool_proj_relu",
            "mixed4b",
            "mixed4b.conv_1x1",
            "mixed4b.conv_1x1_relu",
            "mixed4b.conv_3x3_reduce",
            "mixed4b.conv_3x3_reduce_relu",
            "mixed4b.conv_3x3",
            "mixed4b.conv_3x3_relu",
            "mixed4b.conv_5x5_reduce",
            "mixed4b.conv_5x5_reduce_relu",
            "mixed4b.conv_5x5",
            "mixed4b.conv_5x5_relu",
            "mixed4b.pool",
            "mixed4b.pool_proj",
            "mixed4b.pool_proj_relu",
            "mixed4c",
            "mixed4c.conv_1x1",
            "mixed4c.conv_1x1_relu",
            "mixed4c.conv_3x3_reduce",
            "mixed4c.conv_3x3_reduce_relu",
            "mixed4c.conv_3x3",
            "mixed4c.conv_3x3_relu",
            "mixed4c.conv_5x5_reduce",
            "mixed4c.conv_5x5_reduce_relu",
            "mixed4c.conv_5x5",
            "mixed4c.conv_5x5_relu",
            "mixed4c.pool",
            "mixed4c.pool_proj",
            "mixed4c.pool_proj_relu",
            "mixed4d",
            "mixed4d.conv_1x1",
            "mixed4d.conv_1x1_relu",
            "mixed4d.conv_3x3_reduce",
            "mixed4d.conv_3x3_reduce_relu",
            "mixed4d.conv_3x3",
            "mixed4d.conv_3x3_relu",
            "mixed4d.conv_5x5_reduce",
            "mixed4d.conv_5x5_reduce_relu",
            "mixed4d.conv_5x5",
            "mixed4d.conv_5x5_relu",
            "mixed4d.pool",
            "mixed4d.pool_proj",
            "mixed4d.pool_proj_relu",
            "mixed4e",
            "mixed4e.conv_1x1",
            "mixed4e.conv_1x1_relu",
            "mixed4e.conv_3x3_reduce",
            "mixed4e.conv_3x3_reduce_relu",
            "mixed4e.conv_3x3",
            "mixed4e.conv_3x3_relu",
            "mixed4e.conv_5x5_reduce",
            "mixed4e.conv_5x5_reduce_relu",
            "mixed4e.conv_5x5",
            "mixed4e.conv_5x5_relu",
            "mixed4e.pool",
            "mixed4e.pool_proj",
            "mixed4e.pool_proj_relu",
            "pool4",
            "mixed5a",
            "mixed5a.conv_1x1",
            "mixed5a.conv_1x1_relu",
            "mixed5a.conv_3x3_reduce",
            "mixed5a.conv_3x3_reduce_relu",
            "mixed5a.conv_3x3",
            "mixed5a.conv_3x3_relu",
            "mixed5a.conv_5x5_reduce",
            "mixed5a.conv_5x5_reduce_relu",
            "mixed5a.conv_5x5",
            "mixed5a.conv_5x5_relu",
            "mixed5a.pool",
            "mixed5a.pool_proj",
            "mixed5a.pool_proj_relu",
            "mixed5b",
            "mixed5b.conv_1x1",
            "mixed5b.conv_1x1_relu",
            "mixed5b.conv_3x3_reduce",
            "mixed5b.conv_3x3_reduce_relu",
            "mixed5b.conv_3x3",
            "mixed5b.conv_3x3_relu",
            "mixed5b.conv_5x5_reduce",
            "mixed5b.conv_5x5_reduce_relu",
            "mixed5b.conv_5x5",
            "mixed5b.conv_5x5_relu",
            "mixed5b.pool",
            "mixed5b.pool_proj",
            "mixed5b.pool_proj_relu",
            "avgpool",
            "drop",
            "fc",
        ]
        model = googlenet(pretrained=True)
        collected_layers = model_utils.get_model_layers(model)
        self.assertEqual(collected_layers, expected_list)

    def test_get_layers_torchvision_alexnet(self) -> None:
        try:
            from torchvision import alexnet  # noqa: F401

        except ImportError:
            raise unittest.SkipTest("Skipping alexnet test, torchvision not available.")

        expected_list = [
            "features",
            "features[0]",
            "features[1]",
            "features[2]",
            "features[3]",
            "features[4]",
            "features[5]",
            "features[6]",
            "features[7]",
            "features[8]",
            "features[9]",
            "features[10]",
            "features[11]",
            "features[12]",
            "avgpool",
            "classifier",
            "classifier[0]",
            "classifier[1]",
            "classifier[2]",
            "classifier[3]",
            "classifier[4]",
            "classifier[5]",
            "classifier[6]",
        ]
        model = alexnet(pretrained=True)
        collected_layers = model_utils.get_model_layers(model)
        self.assertEqual(collected_layers, expected_list)


class TestCollectActivations(BaseTest):
    def test_collect_activations(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping collect_activations test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)

        activ_out = model_utils.collect_activations(
            model, [model.mixed4d], torch.zeros(1, 3, 224, 224)
        )

        self.assertIsInstance(activ_out, dict)
        m4d_activ = activ_out[model.mixed4d]
        self.assertEqual(list(m4d_activ.shape), [1, 528, 14, 14])


if __name__ == "__main__":
    unittest.main()
