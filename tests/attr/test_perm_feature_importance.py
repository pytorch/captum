#!/usr/bin/env python3

import random

import torch
import torch.nn as nn
from captum.attr._core.perm_feature_importance import (
    PermutationFeatureImportance,
    permute_feature,
)

from .helpers.basic_models import BasicModel, MultiplyModel2Input
from .helpers.utils import BaseTest


class Test(BaseTest):
    def test_perm_fn(self):
        def check_features_are_permuted(inp, perm_inp, mask):
            unpermuted_features = mask.bitwise_not()
            self.assertTrue((inp[:, mask] != perm_inp[:, mask]).any())
            self.assertTrue(
                (inp[:, unpermuted_features] == perm_inp[:, unpermuted_features]).all()
            )

        sizes_to_test = [(10,), (4, 5), (3, 4, 5), (6, 7, 8, 9)]
        for batch_size in [2, 10]:
            for inp_size in sizes_to_test:
                inp = torch.randn((batch_size,) + inp_size)
                flat_mask = torch.zeros_like(inp[0]).flatten().bool()

                # test random set of single features
                num_features = inp.numel() // batch_size
                num_features_to_test = min(num_features, random.randint(2, 30))
                for _ in range(num_features_to_test):
                    feature_idx = random.randint(0, num_features - 1)
                    flat_mask[feature_idx] = 1

                    # ensure we only set one feature to be masked
                    self.assertTrue(torch.sum(flat_mask) == 1)

                    # permute the feature
                    mask = flat_mask.view_as(inp[0])
                    perm_inp = permute_feature(inp, mask)
                    check_features_are_permuted(inp, perm_inp, mask)

                    flat_mask[feature_idx] = 0

                # test random set of features
                for _ in range(random.randint(10, 20)):
                    mask = torch.zeros_like(inp[0])
                    while mask.sum() == 0:
                        mask = torch.randint_like(inp[0], 0, 2).bool()

                    perm_inp = permute_feature(inp, mask)
                    check_features_are_permuted(inp, perm_inp, mask)

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
        net = MultiplyModel2Input()

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
