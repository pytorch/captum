#!/usr/bin/env python3

import unittest
import copy
import torch
from torch import Tensor
from functools import reduce
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
    BaseGPUTest,
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
)

def _get_nested_attr(obj, layer_name):
    return reduce(getattr, layer_name.split("."), obj)

class TestDP(BaseGPUTest):
    pass

def make_single_test_method(algorithm, model, target_layer, args, target_delta, noise_tunnel, baseline_distr):
    model_cuda = copy.deepcopy(model).cuda()
    dp_model = torch.nn.parallel.DataParallel(model_cuda)
    def _target_batch_test_assert(self):
        if target_layer:
            attr_method = algorithm(model, _get_nested_attr(model, target_layer))
            attr_method_cuda = algorithm(model_cuda, _get_nested_attr(model_cuda, target_layer))
            attr_method_dp = algorithm(dp_model, _get_nested_attr(model_cuda, target_layer))
        else:
            attr_method = algorithm(model)
            attr_method_cuda = algorithm(model_cuda)
            attr_method_dp = algorithm(dp_model)

        if noise_tunnel:
            attr_method = NoiseTunnel(attr_method)
            attr_method_cuda = NoiseTunnel(attr_method_cuda)
            attr_method_dp = NoiseTunnel(attr_method_dp)
        attributions_orig = attr_method.attribute(**args)
        cuda_args = {}
        for key in args:
            if isinstance(args[key], Tensor):
                cuda_args[key] = args[key].cuda()
            elif isinstance(args[key], tuple):
                cuda_args[key] = tuple(elem.cuda if isinstance(elem, Tensor) else elem for elem in args[key])
            else:
                cuda_args[key] = args[key]
        attributions_cuda = attr_method_cuda.attribute(**cuda_args)
        attributions_dp = attr_method_dp.attribute(**cuda_args)
        #assertTensorTuplesAlmostEqual(
        #    self, attributions_cuda, attributions_orig, delta=target_delta
        #)

        assertTensorTuplesAlmostEqual(
            self, attributions_cuda, attributions_dp, delta=target_delta
        )

    return _target_batch_test_assert

# Remaining tests are added below
def make_test_methods(test_config):
    if "skip_dp" in test_config and test_config["skip_dp"] is True:
        return None

    algorithms = test_config["algorithms"]
    model = test_config["model"]
    args = test_config["attribute_args"]
    target_layer = test_config["layer"] if "layer" in test_config else None
    target_delta = test_config["target_delta"] if "target_delta" in test_config else 0.0001
    noise_tunnel = test_config["noise_tunnel"] if "noise_tunnel" in test_config else False
    baseline_distr = test_config["baseline_distr"] if "baseline_distr" in test_config else False

    all_tests = []
    for algorithm in algorithms:
        test_method = make_single_test_method(algorithm, model, target_layer, args, target_delta, noise_tunnel, baseline_distr)
        test_name = algorithm.__name__ + ("NoiseTunnel" if noise_tunnel else "")
        all_tests.append((test_method, "test_dp_" + test_name + "_" + test_config["name"] ))

    return all_tests


for single_test in config:
    test_details = make_test_methods(single_test)
    if test_details is not None:
        for test_func, test_name in test_details:
            setattr(TestDP, test_name, test_func)


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
