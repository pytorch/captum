#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

import captum.optim._utils.models as model_utils
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


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

        self.assertNotIsInstance(toy_model.relu1, old_layer)
        self.assertIsInstance(toy_model.relu1, new_layer)

        self.assertNotIsInstance(toy_model.relu2.relu, old_layer)
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


if __name__ == "__main__":
    unittest.main()
