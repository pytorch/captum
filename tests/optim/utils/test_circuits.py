#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.optim.helpers import numpy_circuits


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


class TestHeatMap(BaseTest):
    def test_heatmap(self) -> None:
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = circuits.tensor_heatmap(x)
        x_out_np = numpy_circuits.array_heatmap(x.numpy())
        assertTensorAlmostEqual(self, x_out, torch.as_tensor(x_out_np).float())


if __name__ == "__main__":
    unittest.main()
