#!/usr/bin/env python3
import unittest

import torch

import captum.optim as opt
from tests.helpers.basic import BaseTest
from tests.helpers.basic_models import BasicModel_ConvNet_Optim


class TestInputOptimization(BaseTest):
    def test_input_optimization(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping InputOptimization test due to insufficient Torch version."
            )
        model = BasicModel_ConvNet_Optim()
        loss_fn = opt.ChannelActivation(model.layer, 0)
        obj = opt.InputOptimization(model, loss_function=loss_fn)
        n_steps = 5
        history = obj.optimize(opt.n_steps(n_steps, show_progress=False))
        self.assertTrue(history[0] > history[-1])
        self.assertTrue(len(history) == n_steps)
