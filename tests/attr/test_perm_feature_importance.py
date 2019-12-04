#!/usr/bin/env python3

import random

import torch
from captum.attr._core.perm_feature_importance import (
    PermutationFeatureImportance,
    _permute_feature,
)

from .helpers.utils import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def _check_features_are_permuted(self, inp, perm_inp, mask):
        permuted_features = mask.expand_as(inp[0])
        unpermuted_features = mask.expand_as(inp[0]).bitwise_not()

        self.assertTrue((inp[:, permuted_features] != perm_inp[:, permuted_features]).any())
        self.assertTrue(
            (inp[:, unpermuted_features] == perm_inp[:, unpermuted_features]).all()
        )

    def _check_perm_fn_with_mask(self, inp, mask):
        perm_inp = _permute_feature(inp, mask)
        self._check_features_are_permuted(inp, perm_inp, mask)

    def test_perm_fn_single_feature(self):
        batch_size = 2
        sizes_to_test = [(10,), (4, 5), (3, 4, 5)]
        for inp_size in sizes_to_test:
            inp = torch.randn((batch_size,) + inp_size)
            flat_mask = torch.zeros_like(inp[0]).flatten().bool()

            num_features = inp.numel() // batch_size
            for i in range(num_features):
                flat_mask[i] = 1
                self._check_perm_fn_with_mask(inp, flat_mask.view_as(inp[0]))
                flat_mask[i] = 0

    def test_perm_fn_broadcastable_masks(self):
        batch_size = 5
        inp_size = (3, 20, 30)

        inp = torch.randn((batch_size,) + inp_size)

        # to be broadcastable dimensions have
        # match from end to beginning, by:
        # 1. Equalling 1
        # 2. Equally the dimension
        #
        # Here I write them explicitly for clarity
        mask_sizes = [
            # dims = 1
            (1, 20, 30),
            (3, 1, 30),
            (3, 20, 1),
            (1, 1, 30),
            (1, 20, 1),

            # missing
            (1,), # empty set (all features)
            (30,),
            (20, 30),
            (3, 20, 30),
        ]

        for mask_size in mask_sizes:
            mask = torch.randint(0, 2, mask_size).bool()
            self.assertTrue(mask.shape == mask_size)

            self._check_perm_fn_with_mask(inp, mask)

    def test_single_input(self):
        batch_size = 2
        input_size = (3,)

        def forward_func(x):
            x = torch.sum(x)
            return torch.sum(x)

        feature_importance = PermutationFeatureImportance(forward_func=forward_func)

        inp = torch.randn((batch_size,) + input_size) * 10

        inp[:, 0] = 5
        for x in range(10):
            attribs = feature_importance.attribute(inp)

            self.assertTrue(attribs.squeeze(0).size() == input_size)
            self.assertTrue((attribs[:, 0] == 0).all())
            self.assertTrue((attribs[:, 1] != 0).all())
            self.assertTrue((attribs[:, 2] != 0).all())

    def test_multi_input(self):
        batch_size = 5
        input_size = (2,)

        labels = torch.randn(batch_size)

        def forward_func(*x):
            y = 0
            for xx in x:
                y += xx[:, 0] * xx[:, 1]
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

    # TODO
    #def test_feature_single(self):
    #    def forward_func(x, y):
    #        return x + y

    #    batch_size = 2
    #    inp1 = torch.randn((batch_size,) + (3,))
    #    inp2 = torch.randn((batch_size,) + (3,))

    #    feature_importance = PermutationFeatureImportance(forward_func=forward_func)

    #    target = 1
    #    mask = (torch.tensor(0), torch.tensor(1))
    #    attribs = feature_importance.attribute(
    #        (inp1, inp2), feature_mask=mask, target=target
    #    )
    #    self.assertTrue(isinstance(attribs, tuple))

    #    features_in_mask = []
    #    for sub_mask in mask:
    #        if sub_mask.numel() == 1:
    #            sub_mask = [sub_mask]

    #        for feature in sub_mask:
    #            features_in_mask.append(feature)

    #    for feature in features_in_mask:
    #        for inp, sub_mask, sub_attrib in zip((inp1, inp2), mask, attribs):
    #            feature_mask = sub_mask == feature
    #            # if the mask doesn't contain feature - skip
    #            if feature_mask.sum() == 0:
    #                continue

    #            # TODO: refactor inp[0] to output space size
    #            feature_mask = feature_mask.expand_as(inp[0])

    #            # if this feature does contribute to the output
    #            if feature_mask[target]:
    #                # two conditions should hold:
    #                # 1) all values in attribs[feature_mask] are the same
    #                # 2) since the batch_size = 2, this value should be
    #                #    equal to (inp[0, target] - inp[1, target],
    #                #              inp[1, target] - inp[0, target])
    #                val = inp[0, target] - inp[1, target]
    #                assertTensorAlmostEqual(self, sub_attrib[0, feature_mask], val)
    #                assertTensorAlmostEqual(self, sub_attrib[1, feature_mask], -val)
    #            else:
    #                # if the feature doesn't contribute to the output
    #                # -- then this means it should have attrib of 0
    #                assertTensorAlmostEqual(self, sub_attrib[:, feature_mask], 0)
        
