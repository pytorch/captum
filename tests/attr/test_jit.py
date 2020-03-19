#!/usr/bin/env python3
import unittest
import copy
import torch
from enum import Enum
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
from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.attr._core.shapley_value import ShapleyValueSampling
from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap

from captum.attr._core.neuron.neuron_conductance import NeuronConductance
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from captum.attr._utils.common import _format_input, _format_additional_forward_args
from .helpers.basic_models import BasicModel_MultiLayer

from .helpers.utils import (
    BaseTest,
    BaseGPUTest,
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    get_nested_attr,
)
from .helpers.test_config import config

JIT_SUPPORTED = [
    IntegratedGradients,
    FeatureAblation,
    FeaturePermutation,
    GradientShap,
    InputXGradient,
    Occlusion,
    Saliency,
    ShapleyValueSampling,
]


class JITCompareMode(Enum):
    """
    Defines modes for DataParallel tests:
    cpu_cuda - Compares results when running attribution method on CPU vs GPU / CUDA
    data_parallel_default - Compares results when running attribution method on GPU with DataParallel
    data_parallel_alt_dev_ids - Compares results when running attribution method on GPU with DataParallel, but with an alternate device ID ordering (not default)
    """

    cpu_jit_trace = 1
    cpu_jit_script = 2
    data_parallel_jit_trace = 3
    data_parallel_jit_script = 3


class JITMeta(type):
    def __new__(cls, name, bases, attrs):
        for test_config in config:
            algorithms = test_config["algorithms"]
            model = test_config["model"]
            args = test_config["attribute_args"]
            target_delta = (
                test_config["target_delta"] if "target_delta" in test_config else 0.0002
            )
            noise_tunnel = (
                test_config["noise_tunnel"] if "noise_tunnel" in test_config else False
            )
            baseline_distr = (
                test_config["baseline_distr"]
                if "baseline_distr" in test_config
                else False
            )

            for algorithm in algorithms:
                if algorithm in JIT_SUPPORTED:
                    for mode in JITCompareMode:
                        # Creates test case corresponding to each algorithm and DataParallelCompareMode
                        test_method = cls.make_single_jit_test(
                            algorithm,
                            model,
                            args,
                            target_delta,
                            noise_tunnel,
                            baseline_distr,
                            mode,
                        )
                        test_name = (
                            "test_"
                            + test_config["name"]
                            + "_"
                            + mode.name
                            + "_"
                            + algorithm.__name__
                            + ("NoiseTunnel" if noise_tunnel else "")
                        )
                        attrs[test_name] = test_method

        return super(JITMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def make_single_jit_test(
        cls, algorithm, model, args, target_delta, noise_tunnel, baseline_distr, mode
    ):
        """
        This method creates a single Data Parallel / GPU test for the given algorithm and parameters.
        """
        model_1 = model
        # Construct cuda_args, moving all tensor inputs in args to CUDA device
        if (
            mode is JITCompareMode.data_parallel_jit_trace
            or JITCompareMode.data_parallel_jit_script
        ):
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                raise unittest.SkipTest("Skipping GPU test since CUDA not available.")

            cuda_args = {}
            for key in args:
                if isinstance(args[key], Tensor):
                    cuda_args[key] = args[key].cuda()
                elif isinstance(args[key], tuple):
                    cuda_args[key] = tuple(
                        elem.cuda() if isinstance(elem, Tensor) else elem
                        for elem in args[key]
                    )
                else:
                    cuda_args[key] = args[key]
            args = cuda_args
            model_1 = copy.deepcopy(model).cuda()

        # Initialize models based on DataParallelCompareMode
        if (
            mode is JITCompareMode.cpu_jit_script
            or JITCompareMode.data_parallel_jit_script
        ):
            model_2 = torch.jit.script(model_1)
        elif (
            mode is JITCompareMode.cpu_jit_trace
            or JITCompareMode.data_parallel_jit_trace
        ):
            all_inps = _format_input(args["inputs"]) + (
                _format_additional_forward_args(args["additional_forward_args"])
                if "additional_forward_args" in args
                and args["additional_forward_args"] is not None
                else tuple()
            )
            model_2 = torch.jit.trace(model_1, all_inps)
        else:
            raise AssertionError("JIT compare mode type is not valid.")

        def jit_test_assert(self):
            attr_method_1 = algorithm(model_1)
            attr_method_2 = algorithm(model_2)

            if noise_tunnel:
                attr_method_1 = NoiseTunnel(attr_method_1)
                attr_method_2 = NoiseTunnel(attr_method_2)
            if attr_method_1.has_convergence_delta():
                attributions_1, delta_1 = attr_method_1.attribute(
                    return_convergence_delta=True, **args
                )
                self.setUp()
                attributions_2, delta_2 = attr_method_2.attribute(
                    return_convergence_delta=True, **args
                )
                assertTensorTuplesAlmostEqual(
                    self, attributions_1, attributions_2, delta=target_delta, mode="max"
                )
                assertTensorTuplesAlmostEqual(
                    self, delta_1, delta_2, delta=target_delta, mode="max"
                )
            else:
                attributions_1 = attr_method_1.attribute(**args)
                self.setUp()
                attributions_2 = attr_method_2.attribute(**args)
                assertTensorTuplesAlmostEqual(
                    self, attributions_1, attributions_2, delta=target_delta, mode="max"
                )

        return jit_test_assert


class JITTest(BaseTest, metaclass=JITMeta):
    pass
