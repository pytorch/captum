#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest


class TestActivationCatcher(BaseTest):
    def test_activation_catcher(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ActivationCatcher test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)
        try:
            catch_activ = circuits.ActivationCatcher(targets=[model.mixed4d])
            activ_out = catch_activ(model, torch.zeros(1, 3, 224, 224))
            self.assertIsInstance(activ_out, dict)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)


class TestGetExpandedWeights(BaseTest):
    def test_get_expanded_weights(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)
        output_tensor = circuits.get_expanded_weights(
            model, model.mixed4c, model.mixed4d
        )
        self.assertTrue(torch.is_tensor(output_tensor))


if __name__ == "__main__":
    unittest.main()
