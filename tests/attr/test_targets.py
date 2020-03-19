#!/usr/bin/env python3


import torch
from torch import Tensor

from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _format_additional_forward_args

from .helpers.basic_models import BasicModel_MultiLayer
from .helpers.test_config import config
from .helpers.utils import (
    BaseTest,
    assertTensorTuplesAlmostEqual,
    deep_copy_args,
    get_nested_attr,
)


class TargetsMeta(type):
    def __new__(cls, name, bases, attrs):
        for test_config in config:
            algorithms = test_config["algorithms"]
            model = test_config["model"]
            args = test_config["attribute_args"]
            layer = test_config["layer"] if "layer" in test_config else None
            target_delta = (
                test_config["target_delta"] if "target_delta" in test_config else 0.0001
            )
            noise_tunnel = (
                test_config["noise_tunnel"] if "noise_tunnel" in test_config else False
            )
            baseline_distr = (
                test_config["baseline_distr"]
                if "baseline_distr" in test_config
                else False
            )

            if "target" not in args or not isinstance(args["target"], (list, Tensor)):
                continue

            for algorithm in algorithms:
                if issubclass(algorithm, FeaturePermutation):
                    continue
                test_method = cls.make_single_target_test(
                    algorithm,
                    model,
                    layer,
                    args,
                    target_delta,
                    noise_tunnel,
                    baseline_distr,
                )
                test_name = (
                    "test_target_"
                    + test_config["name"]
                    + "_"
                    + algorithm.__name__
                    + ("NoiseTunnel" if noise_tunnel else "")
                )
                attrs[test_name] = test_method
        return super(TargetsMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    @deep_copy_args
    def make_single_target_test(
        cls, algorithm, model, layer, args, target_delta, noise_tunnel, baseline_distr
    ):
        """
        This method creates a single target test for the given algorithm and parameters.
        """

        target_layer = get_nested_attr(model, layer) if layer is not None else None
        original_inputs = args["inputs"]
        original_targets = args["target"]
        original_additional_forward_args = (
            _format_additional_forward_args(args["additional_forward_args"])
            if "additional_forward_args" in args
            else None
        )
        num_examples = (
            len(original_inputs)
            if isinstance(original_inputs, Tensor)
            else len(original_inputs[0])
        )
        replace_baselines = "baselines" in args and not baseline_distr
        if replace_baselines:
            original_baselines = args["baselines"]

        def target_test_assert(self):
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
                    else tuple(
                        original_inp[i : i + 1] for original_inp in original_inputs
                    )
                )
                if original_additional_forward_args is not None:
                    args["additional_forward_args"] = tuple(
                        single_add_arg[i : i + 1]
                        if isinstance(single_add_arg, Tensor)
                        else single_add_arg
                        for single_add_arg in original_additional_forward_args
                    )
                if replace_baselines:
                    if isinstance(original_inputs, Tensor):
                        args["baselines"] = original_baselines[i : i + 1]
                    elif isinstance(original_baselines, tuple):
                        args["baselines"] = tuple(
                            single_baseline[i : i + 1]
                            if isinstance(single_baseline, Tensor)
                            else single_baseline
                            for single_baseline in original_baselines
                        )
                self.setUp()
                single_attr = attr_method.attribute(**args)
                current_orig_attributions = (
                    attributions_orig[i : i + 1]
                    if isinstance(attributions_orig, Tensor)
                    else tuple(
                        single_attrib[i : i + 1] for single_attrib in attributions_orig
                    )
                )
                assertTensorTuplesAlmostEqual(
                    self,
                    current_orig_attributions,
                    single_attr,
                    delta=target_delta,
                    mode="max",
                )
                if len(original_targets) == num_examples:
                    args["target"] = original_targets[i : i + 1]
                    self.setUp()
                    single_attr_target_list = attr_method.attribute(**args)
                    assertTensorTuplesAlmostEqual(
                        self,
                        current_orig_attributions,
                        single_attr_target_list,
                        delta=target_delta,
                        mode="max",
                    )

        return target_test_assert


class TestTargets(BaseTest, metaclass=TargetsMeta):
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
