#!/usr/bin/env python3
import unittest

import torch

import captum.optim as opt
from tests.helpers.basic import BaseTest
from tests.helpers.basic_models import BasicModel_ConvNet_Optim


class TestInputOptimization(BaseTest):
    @unittest.skipIf(torch.__version__ > "1.8.1", "Bug in PyTorch nightly build")
    def test_input_optimization(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping InputOptimization test due to insufficient Torch version."
            )
        model = BasicModel_ConvNet_Optim()
        loss_fn = opt.loss.ChannelActivation(model.layer, 0)
        obj = opt.InputOptimization(model, loss_function=loss_fn)
        n_steps = 5
        history = obj.optimize(opt.optimization.n_steps(n_steps, show_progress=False))
        self.assertTrue(history[0] > history[-1])
        self.assertTrue(len(history) == n_steps)

    @unittest.skipIf(torch.__version__ > "1.8.1", "Bug in PyTorch nightly build")
    def test_input_optimization_param(self) -> None:
        """Test for optimizing param without model"""
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping InputOptimization test due to insufficient Torch version."
            )
        img_param = opt.images.NaturalImage()
        loss_fn = opt.loss.ChannelActivation(img_param, 0)
        # Use torch.nn.Identity as placeholder for non-model optimization
        obj = opt.InputOptimization(torch.nn.Identity(), loss_fn, img_param)
        n_steps = 5
        history = obj.optimize(opt.optimization.n_steps(n_steps, show_progress=False))
        self.assertTrue(history[0] > history[-1])
        self.assertTrue(len(history) == n_steps)
