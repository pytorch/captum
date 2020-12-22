#!/usr/bin/env python3
import unittest

import torch

import captum.optim._core.output_hook as output_hook
from tests.helpers.basic import BaseTest


class TestActivationFetcher(BaseTest):
    def test_activation_fetcher(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ActivationFetcher test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)
        try:
            catch_activ = output_hook.ActivationFetcher(model, targets=[model.mixed4d])
            activ_out = catch_activ(torch.zeros(1, 3, 224, 224))
            self.assertIsInstance(activ_out, dict)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()
