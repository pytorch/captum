#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from captum.optim._utils.models import RedirectedReluLayer, ignore_layer, max2avg_pool2d
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

    def test_get_expanded_nonlinear_top_connections(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights nonlinear_top_connections test"
                + " due to insufficient Torch version."
            )

        if torch.__version__ == "1.3.0":
            norm_func = torch.norm
        else:
            norm_func = torch.linalg.norm
        model = googlenet(pretrained=True)
        max2avg_pool2d(model)
        ignore_layer(model, RedirectedReluLayer)
        output_tensor = circuits.get_expanded_weights(
            model, model.pool3, model.mixed4a, 5
        )
        self.assertEqual(list(output_tensor.shape), [508, 480, 5, 5])

        top_connected_neurons = torch.argsort(
            torch.stack(
                [
                    -norm_func(output_tensor[i, 379, :, :])
                    for i in range(output_tensor.shape[0])
                ]
            )
        )[:10].tolist()

        expected_list = [50, 437, 96, 398, 434, 423, 408, 436, 424, 168]
        self.assertEqual(top_connected_neurons, expected_list)


if __name__ == "__main__":
    unittest.main()
