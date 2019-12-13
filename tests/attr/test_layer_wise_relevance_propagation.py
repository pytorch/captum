#!/usr/bin/env python3

import torch
import torch.nn as nn
import unittest
#from unittest.TestCase import assertRaises
from captum.attr._core.layer_wise_relevance_propagation import LRP, LRP_0, EpsilonRhoRule, BasicRule

from .helpers.basic_models import BasicModel_ConvNet_One_Conv
from .helpers.utils import BaseTest, assertTensorAlmostEqual
import math

def _get_basic_config():
    input = torch.arange(16).view(1, 1, 4, 4).float()
    grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
    expected_activations = torch.tensor([[54.1300, 54.1880, 54.2934, 53.9288]])
    return BasicModel_ConvNet_One_Conv(), input, grads, expected_activations, None

def _get_rule_config():
    relevance = torch.tensor([[[-0.0, 3.0]]])
    layer = nn.modules.Conv1d(1, 1, 2, bias=False)
    nn.init.constant_(layer.weight.data, 2)
    activations = torch.tensor([[[1.0, 5.0, 7.0]]])
    return relevance, layer, activations


class Test(BaseTest):
    def test_lrp_test_activations(self):
        self._activations_base_assert(*_get_basic_config())

    def test_lrp_creator(self):
        self._creator_error_assert(*_get_basic_config())

    def test_basic_rule(self):
        self._rule_assert(*_get_rule_config(), rule="Basic")

    def test_basic_rule_init(self):
        self._rule_assert(*_get_rule_config(), rule="BasicInit")

    def _activations_base_assert(self, model, inputs, grads, expected_activations, expected):
        lrp = LRP_0(model)
        activations = lrp._get_activations(inputs)
        assertTensorAlmostEqual(self, activations[3], expected_activations)

    def _creator_error_assert(self, model, inputs, grads, expected_activations, expected):
        self.assertRaises(AssertionError, LRP, model, [5])

    def _rule_assert(self, relevance, layer, activations, rule):
        if rule == "Basic":
            basic = BasicRule()
            propagatedRelevance = basic.propagate(relevance, layer, activations)
            expectedRelevance = torch.tensor([0.0, 1.25, 1.75])
            assertTensorAlmostEqual(self, expectedRelevance, propagatedRelevance)
        elif rule == "BasicInit":
            basic = BasicRule()
            self.assertEqual(basic._epsilon(), 0)