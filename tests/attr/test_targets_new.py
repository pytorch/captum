#!/usr/bin/env python3

import unittest

import torch
from torch import Tensor
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.saliency import Saliency
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.occlusion import Occlusion

from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap

from captum.attr._core.neuron.neuron_conductance import NeuronConductance

from .helpers.basic_models import BasicModel_MultiLayer
from .helpers.test_config import config
from .helpers.utils import (
    BaseTest,
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
)


class TestTargets(BaseTest):
    def test_simple_target_missing_error(self):
        net = BasicModel_MultiLayer()
        inp = torch.zeros((1, 3))
        with self.assertRaises(AssertionError):
            attr = IntegratedGradients(net)
            attr.attribute(inp)

    def test_multi_target_error(self):
        net = BasicModel_MultiLayer()
        inp = torch.zeros((1, 3))
        with self.assertRaises(AssertionError):
            attr = IntegratedGradients(net)
            attr.attribute(inp, additional_forward_args=(None, True), target=(1, 0))


def make_single_test_method(algorithm, model, target_layer, args, target_delta, noise_tunnel, baseline_distr):
    original_inputs = args["inputs"]
    original_targets = args["target"]
    num_examples = (
        len(original_inputs)
        if isinstance(original_inputs, Tensor)
        else len(original_inputs[0])
    )
    replace_baselines = "baselines" in args and not baseline_distr
    if replace_baselines:
        original_baselines = args["baselines"]

    def _target_batch_test_assert(self):
        if target_layer:
            attr_method = algorithm(model, target_layer)
        else:
            attr_method = algorithm(model)

        if noise_tunnel:
            attr_method = NoiseTunnel(attr_method)

        attributions_orig = attr_method.attribute(**args)
        for i in range(num_examples):
            args["target"] = (
                original_targets[i]
                if len(original_targets) == num_examples
                else original_targets
            )
            args["inputs"] = (
                original_inputs[i : i + 1]
                if isinstance(original_inputs, Tensor)
                else tuple(original_inp[i : i + 1] for original_inp in original_inputs)
            )
            if replace_baselines:
                if isinstance(original_inputs, Tensor):
                    args["baselines"] = original_baselines[i : i + 1]
                elif isinstance(original_baselines, tuple):
                    args["baselines"] = tuple(single_baseline[i : i + 1] if isinstance(single_baseline, Tensor) else single_baseline for single_baseline in original_baselines)
            single_attr = attr_method.attribute(**args)
            current_orig_attributions = (
                    attributions_orig[i : i + 1]
                    if isinstance(attributions_orig, Tensor)
                    else tuple(
                        single_attrib[i : i + 1] for single_attrib in attributions_orig
                    )
                )    
            assertTensorTuplesAlmostEqual(
                self, current_orig_attributions, single_attr, delta=target_delta
            )
            if len(original_targets) == num_examples:
                args["target"] = original_targets[i : i + 1]
                single_attr_target_list = attr_method.attribute(**args)
                
                assertTensorTuplesAlmostEqual(
                    self,
                    current_orig_attributions,
                    single_attr_target_list,
                    delta=target_delta,
                )
        args["inputs"] = original_inputs
        args["target"] = original_targets
        if replace_baselines:
            args["baselines"] = original_baselines
    return _target_batch_test_assert

# Remaining tests are added below
def make_test_methods(test_config):
    if "skip_targets" in test_config and test_config["skip_targets"] is True:
        return None

    algorithms = test_config["algorithms"]
    model = test_config["model"]
    args = test_config["attribute_args"]
    target_layer = getattr(model, test_config["layer"]) if "layer" in test_config else None
    target_delta = test_config["target_delta"] if "target_delta" in test_config else 0.0001
    noise_tunnel = test_config["noise_tunnel"] if "noise_tunnel" in test_config else False
    baseline_distr = test_config["baseline_distr"] if "baseline_distr" in test_config else False

    if "target" not in args or not isinstance(args["target"], (list, Tensor)):
        return None

    all_tests = []
    for algorithm in algorithms:
        test_method = make_single_test_method(algorithm, model, target_layer, args, target_delta, noise_tunnel, baseline_distr)
        test_name = algorithm.__name__ + ("NoiseTunnel" if noise_tunnel else "")
        all_tests.append((test_method, "test_target_" + test_name + "_" + test_config["name"] ))

    return all_tests


for single_test in config:
    test_details = make_test_methods(single_test)
    if test_details is not None:
        for test_func, test_name in test_details:
            setattr(TestTargets, test_name, test_func)


"""if __name__ == "__main__":
    tests_config = [{"name": "integrated_gradients_basic", "algorithm": IntegratedGradients, "model": BasicModel_MultiLayer(), "args":{"inputs":torch.randn(4, 3), "target":[0,1,1,0]}}]

    for single_test in tests_config:
        test_func, test_name = make_test_function(name, params[0], params[1])
        print(test_name)
        print(test_func)
        if test_func is not None:
            setattr(TestTargets, test_name, test_func)

    unittest.main()
"""
