#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
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


if __name__ == "__main__":
    unittest.main()
