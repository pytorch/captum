#!/usr/bin/env python3

import random

import torch
import torch.nn as nn
from captum.attr._core.perm_feature_importance import (
    PermutationFeatureImportance,
    permute_feature,
)

from .helpers.basic_models import BasicModel, BasicModel_MultiLayer, MultiplyModel2Input
from .helpers.utils import BaseTest, assertTensorAlmostEqual


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

        feature_importance = PermutationFeatureImportance(forward_func=forward_func)

        inp = torch.randn((batch_size,) + input_size)

        inp[:, 0] = 5
        attribs = feature_importance.attribute(inp)
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

        feature_importance = PermutationFeatureImportance(forward_func=forward_func)

        inp = (
            torch.randn((batch_size,) + input_size),
            torch.randn((batch_size,) + input_size),
        )
        inp[1][:, 1] = 4
        attribs = feature_importance.attribute(inp)

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
        batch_size = 2
        input_size = (4,)

        inp = torch.randn((batch_size,) + input_size)

        def forward_func(x):
            return 1 - x

        target = 1
        feature_importance = PermutationFeatureImportance(forward_func=forward_func)

        attribs = feature_importance.attribute(
            inp, ablations_per_eval=ablations_per_eval, target=target
        )
        self.assertTrue(attribs.size() == (batch_size,) + input_size)

        for i in range(inp.size(1)):
            if i == target:
                continue
            assertTensorAlmostEqual(self, attribs[:, i], 0)

        y = forward_func(inp)
        actual_diff = torch.stack([(y[0] - y[1])[target], (y[1] - y[0])[target]])
        assertTensorAlmostEqual(self, attribs[:, target], actual_diff)

    def test_broadcastable_masks(self):
        def forward_func(x, y):
            return x + y

        batch_size = 2
        inp1 = torch.randn((batch_size,) + (3,))
        inp2 = torch.randn((batch_size,) + (3,))
        masks_to_test = [
            (torch.tensor(0), torch.tensor(1)),
            (torch.tensor(0), torch.tensor([0, 0, 1])),
            (torch.tensor([0, 0, 1]), torch.tensor([1, 1, 2])),
        ]
        feature_importance = PermutationFeatureImportance(forward_func=forward_func)
        for masks in masks_to_test:
            target = random.randint(0, 2)
            attribs = feature_importance.attribute(
                (inp1, inp2), feature_mask=masks, target=target
            )
            self.assertTrue(isinstance(attribs, tuple))

            target_mask = torch.zeros_like(inp1[0]).byte()
            target_mask[target] = 1

            features_in_mask = set()
            for mask in masks:
                fvs = mask
                if mask.numel() == 1:
                    fvs = [mask]
                for fv in fvs:
                    features_in_mask.add(fv)

            for feature in features_in_mask:
                for inp, sub_mask, sub_attrib in zip((inp1, inp2), masks, attribs):
                    feature_mask = sub_mask == feature
                    # if the mask doesn't contain feature - skip
                    if feature_mask.sum() == 0:
                        continue

                    feature_mask = feature_mask.expand_as(inp[0])

                    # if this feature does contribute to the output
                    if feature_mask[target]:
                        # two conditions should hold:
                        # 1) all values in attribs[feature_mask] are the same
                        # 2) since the batch_size = 2, this value should be
                        #    equal to (inp[0, target] - inp[1, target], inp[1, target] - inp[0, target])
                        val = inp[0, target] - inp[1, target]
                        assertTensorAlmostEqual(self, sub_attrib[0, feature_mask], val)
                        assertTensorAlmostEqual(self, sub_attrib[1, feature_mask], -val)
                    else:
                        # if the feature doesn't contribute to the output
                        # -- then this means it should have attrib of 0
                        assertTensorAlmostEqual(self, sub_attrib[:, feature_mask], 0)
