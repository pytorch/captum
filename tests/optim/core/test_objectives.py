#!/usr/bin/env python3
import unittest
from typing import cast
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn

from captum.optim._core.output_hook import AbortForwardException, ModuleOutputsHook
import captum.optim as opt
from tests.helpers.basic import BaseTest, assertArraysAlmostEqual


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Conv2d(3, 2, 1)
    def forward(self, x):
        return self.layer(x)


class TestInputOptimization(BaseTest):
    def test_input_optimization(self) -> None:
        model = SimpleModel()
        loss_fn = opt.loss.ChannelActivation(model.layer, 0)
        obj = opt.InputOptimization(model, loss_function=loss_fn)
        history = obj.optimize(opt.objectives.n_steps(128))


if __name__ == "__main__":
    unittest.main()
