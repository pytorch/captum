#!/usr/bin/env python3
import unittest

import torch

import captum.optim._core.output_hook as output_hook
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest


class TestActivationFetcher(BaseTest):
    def test_activation_fetcher(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ActivationFetcher test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)

        catch_activ = output_hook.ActivationFetcher(model, targets=[model.mixed4d])
        activ_out = catch_activ(torch.zeros(1, 3, 224, 224))

        self.assertIsInstance(activ_out, dict)
        m4d_activ = activ_out[model.mixed4d]
        self.assertEqual(list(m4d_activ.shape), [1, 528, 14, 14])


if __name__ == "__main__":
    unittest.main()
