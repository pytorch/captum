#!/usr/bin/env python3

import torch
import torch.nn as nn
from captum.attr._core.perm_feature_importance import (
    PermutationFeatureImportance,
    permute_feature,
)

from .helpers.basic_models import BasicModel
from .helpers.utils import BaseTest


class MultiplyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x[:, 0] * x[:, 1] + y[:, 0] * y[:, 1]


class Test(BaseTest):
    def test_perm_fn(self):
        for bs in [2, 10, 100]:
            inp = torch.randn(bs, 50)

            for i in range(inp.size(1)):
                fm = torch.zeros_like(inp[0])
                fm[i] = 1
                fm = fm.bool()

                perm_inp = permute_feature(inp, fm)

                self.assertTrue((perm_inp[:, i] != inp[:, i]).any())
                for j in range(inp.size(1)):
                    if i == j:
                        continue

                    self.assertTrue((perm_inp[:, j] == inp[:, j]).all())

    def test_single_input(self):
        batch_size = 30
        input_size = (3,)
        net = BasicModel()

        def forward_func(x):
            return torch.sum(net(x))

        fi = PermutationFeatureImportance(forward_func=forward_func)

        inp = torch.randn((batch_size,) + input_size)

        inp[:, 0] = 5
        attribs = fi.attribute(inp)
        self.assertTrue(attribs.squeeze(0).size() == input_size)
        self.assertTrue((attribs[:, 0] == 0).all())
        self.assertTrue((attribs[:, 1] != 0).all())
        self.assertTrue((attribs[:, 2] != 0).all())

    def test_multi_input(self):
        batch_size = 5
        input_size = (2,)
        net = MultiplyNet()

        labels = torch.randn(batch_size)

        def forward_func(*x):
            y = net(*x)
            return torch.sum((y - labels) ** 2)

        fi = PermutationFeatureImportance(forward_func=forward_func)

        inp = (
            torch.randn((batch_size,) + input_size),
            torch.randn((batch_size,) + input_size),
        )
        inp[1][:, 1] = 4
        attribs = fi.attribute(inp)

        self.assertTrue(isinstance(attribs, tuple))
        self.assertTrue(len(attribs) == 2)
        self.assertTrue(attribs[0].squeeze(0).size() == input_size)
        self.assertTrue(attribs[1].squeeze(0).size() == input_size)

        for i in range(2):
            self.assertTrue((attribs[0][:, i] != 0).all())
            if i != 1:
                self.assertTrue((attribs[1][:, i] != 0).all())

        self.assertTrue((attribs[1][:, 1] == 0).all())

    def test_mulitple_ablations(self):
        ablations_per_eval = 4
        batch_size = 25
        input_size = (4,)

        net = BasicModel()

        inp = torch.randn((batch_size,) + input_size)

        def forward_func(x):
            if x.size(0) > batch_size:
                self.assertTrue(ablations_per_eval * batch_size == x.size(0))

                inputs = [
                    x[i * batch_size : (i + 1) * batch_size]
                    for i in range(ablations_per_eval)
                ]
                for i in range(ablations_per_eval):
                    for j in range(ablations_per_eval):
                        if i == j:
                            continue
                        self.assertTrue((inputs[i] != inputs[j]).any())

            y = net(x)
            return y

        fi = PermutationFeatureImportance(forward_func=forward_func)

        attribs = fi.attribute(inp, ablations_per_eval=ablations_per_eval, target=1)
        self.assertTrue(attribs.size() == (batch_size,) + input_size)
