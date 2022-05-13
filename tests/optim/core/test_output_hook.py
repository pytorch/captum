#!/usr/bin/env python3
from typing import cast

import captum.optim._core.output_hook as output_hook
import torch
from captum.optim.models import googlenet
from tests.helpers.basic import BaseTest


class TestActivationFetcher(BaseTest):
    def test_activation_fetcher(self) -> None:
        model = googlenet(pretrained=True)

        catch_activ = output_hook.ActivationFetcher(model, targets=[model.mixed4d])
        activ_out = catch_activ(torch.zeros(1, 3, 224, 224))

        self.assertIsInstance(activ_out, dict)
        m4d_activ = activ_out[model.mixed4d]
        self.assertEqual(list(cast(torch.Tensor, m4d_activ).shape), [1, 528, 14, 14])
