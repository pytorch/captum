#!/usr/bin/env python3

import captum
import torch
from tests.helpers.basic import BaseTest


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class TinyAE(torch.nn.Module):
    def __init__(self):
        super(TinyAE, self).__init__()

        self.linear1 = torch.nn.Linear(100, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 100)

    def encode(self, x):
        x = self.linear1(x)
        return self.activation(x)

    def decode(self, z):
        return self.linear2(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class Test(BaseTest):
    def test_basic_setup(self):

        model = TinyModel()
        ae = TinyAE()
        x = torch.randn(1, 100)

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)

        # Computes counterfactual for class 3.
        output = attr.attribute(x, target=3, lambda_sweep_steps=10)

        assert 10 == len(output["generated_images"])
        assert (1, 100) == output["generated_images"][0].shape
