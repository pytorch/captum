#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest


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


if __name__ == "__main__":
    unittest.main()
