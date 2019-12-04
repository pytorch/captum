#!/usr/bin/env python3

import random

import torch
from captum.attr._core.perm_feature_importance import (
    PermutationFeatureImportance,
    _permute_feature,
)

from .helpers.utils import BaseTest, assertArraysAlmostEqual, assertTensorAlmostEqual
from .helpers.basic_models import BasicModel_ConvNet_One_Conv


class Test(BaseTest):
    def _check_features_are_permuted(self, inp, perm_inp, mask):
        permuted_features = mask.expand_as(inp[0])
        unpermuted_features = mask.expand_as(inp[0]).bitwise_not()

        self.assertTrue(inp.dtype == perm_inp.dtype)
        self.assertTrue(inp.shape == perm_inp.shape)
        self.assertTrue(
            (inp[:, permuted_features] != perm_inp[:, permuted_features]).any()
        )
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

        # To be broadcastable dimensions have
        # match from end to beginning, by equalling 1 or the dim.
        #
        # If a dimension is missing then it must be the
        # last dim provided (from right to left). The missing
        # dimensions are implied to be = 1
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
            (1,),  # empty set (all features)
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
            return x.sum()

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

    def test_mulitple_ablations_per_eval(self):
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
        # integration test to ensure that
        # permutation function works with custom masks
        def forward_func(x):
            return x.view(x.shape[0], -1).sum(dim=-1)

        batch_size = 2
        inp = torch.randn((batch_size,) + (3, 4, 4))

        feature_importance = PermutationFeatureImportance(forward_func=forward_func)

        masks = [
            torch.tensor([0]),
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[[0, 1, 2, 3], [3, 3, 4, 5], [6, 6, 4, 6], [7, 8, 9, 10]]]),
        ]

        for mask in masks:
            attribs = feature_importance.attribute(inp, feature_mask=mask)

            self.assertTrue(attribs is not None)
            self.assertTrue(attribs.shape == inp.shape)

            y = forward_func(inp)
            fm = mask.expand_as(inp[0])

            features = set([x for x in mask.flatten()])
            for feature in features:
                m = (fm == feature).bool()
                attribs_for_feature = attribs[:, m]
                assertArraysAlmostEqual(attribs_for_feature[0], -attribs_for_feature[1])
