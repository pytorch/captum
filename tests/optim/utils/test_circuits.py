#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from captum.optim._utils.models import RedirectedReluLayer
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestGetExpandedWeights(BaseTest):
    def test_get_expanded_weights(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)
        output_tensor = circuits.get_expanded_weights(
            model, model.mixed3a, model.mixed3b
        )
        self.assertTrue(torch.is_tensor(output_tensor))
        self.assertEqual(list(output_tensor.shape), [480, 256, 28, 28])

    def test_get_expanded_weights_crop_int(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights crop test due to insufficient Torch"
                + " version."
            )
        model = googlenet(pretrained=True)
        output_tensor = circuits.get_expanded_weights(
            model, model.mixed3a, model.mixed3b, 5
        )
        self.assertEqual(list(output_tensor.shape), [480, 256, 5, 5])

    def test_get_expanded_weights_crop_two_int(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights two int crop test due to insufficient"
                + " Torch version."
            )
        model = googlenet(pretrained=True)
        output_tensor = circuits.get_expanded_weights(
            model, model.mixed3a, model.mixed3b, (5, 5)
        )
        self.assertEqual(list(output_tensor.shape), [480, 256, 5, 5])

    def test_get_expanded_nonlinear_top_connections(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights nonlinear_top_connections test"
                + " due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)
        circuits.max2avg_pool2d(model)
        circuits.ignore_layer(model, RedirectedReluLayer)
        output_tensor = circuits.get_expanded_weights(
            model, model.pool3, model.mixed4a, 5
        )
        self.assertEqual(list(output_tensor.shape), [508, 480, 5, 5])

        top_connected_neurons = torch.argsort(
            torch.stack(
                [
                    -torch.linalg.norm(output_tensor[i, 379, :, :])
                    for i in range(output_tensor.shape[0])
                ]
            )
        )[:10].tolist()

        expected_list = [50, 437, 96, 398, 434, 423, 436, 168, 408, 415]
        self.assertEqual(top_connected_neurons, expected_list)


class TestMax2AvgPool2d(BaseTest):
    def test_max2avg_pool2d(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

        circuits.max2avg_pool2d(model)

        test_tensor = torch.randn(128, 32, 16, 16)
        test_tensor = F.pad(test_tensor, (0, 1, 0, 1), value=float("-inf"))
        out_tensor = model(test_tensor)

        avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        expected_tensor = avg_pool(test_tensor)
        expected_tensor[expected_tensor == float("-inf")] = 0.0

        assertTensorAlmostEqual(self, out_tensor, expected_tensor, 0)


class TestIgnoreLayer(BaseTest):
    def test_ignore_layer(self) -> None:
        model = torch.nn.Sequential(torch.nn.ReLU())
        x = torch.randn(1, 3, 4, 4)
        circuits.ignore_layer(model, torch.nn.ReLU)
        output_tensor = model(x)
        assertTensorAlmostEqual(self, x, output_tensor, 0)


if __name__ == "__main__":
    unittest.main()
