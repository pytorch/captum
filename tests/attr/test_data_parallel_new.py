#!/usr/bin/env python3
import copy
import torch
from enum import Enum
from torch import Tensor
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.guided_grad_cam import GuidedGradCam

from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap

from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap

from .helpers.utils import (
    BaseGPUTest,
    assertTensorTuplesAlmostEqual,
    get_nested_attr,
)
from .helpers.test_config import config


class DataParallelCompareMode(Enum):
    """
    Defines modes for DataParallel tests:
    cpu_cuda - Compares results when running attribution method on CPU vs GPU / CUDA
    data_parallel_default - Compares results when running attribution method on GPU with
    DataParallel
    data_parallel_alt_dev_ids - Compares results when running attribution method on GPU
    with DataParallel, but with an alternate device ID ordering (not default)
    """

    cpu_cuda = 1
    data_parallel_default = 2
    data_parallel_alt_dev_ids = 3


class DataParallelMeta(type):
    def __new__(cls, name, bases, attrs):
        for test_config in config:
            algorithms = test_config["algorithms"]
            model = test_config["model"]
            args = test_config["attribute_args"]
            target_layer = test_config["layer"] if "layer" in test_config else None
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
                for mode in DataParallelCompareMode:
                    # Creates test case corresponding to each algorithm and
                    # DataParallelCompareMode
                    test_method = cls.make_single_dp_test(
                        algorithm,
                        model,
                        target_layer,
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

        return super(DataParallelMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def make_single_dp_test(
        cls,
        algorithm,
        model,
        target_layer,
        args,
        target_delta,
        noise_tunnel,
        baseline_distr,
        mode,
    ):
        """
        This method creates a single Data Parallel / GPU test for the given
        algorithm and parameters.
        """
        # Construct cuda_args, moving all tensor inputs in args to CUDA device
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

        alt_device_ids = None
        cuda_model = copy.deepcopy(model).cuda()
        # Initialize models based on DataParallelCompareMode
        if mode is DataParallelCompareMode.cpu_cuda:
            model_1, model_2 = model, cuda_model
            args_1, args_2 = args, cuda_args
        elif mode is DataParallelCompareMode.data_parallel_default:
            model_1, model_2 = cuda_model, torch.nn.parallel.DataParallel(cuda_model)
            args_1, args_2 = cuda_args, cuda_args
        elif mode is DataParallelCompareMode.data_parallel_alt_dev_ids:
            alt_device_ids = [0] + [
                x for x in range(torch.cuda.device_count() - 1, 0, -1)
            ]
            model_1, model_2 = (
                cuda_model,
                torch.nn.parallel.DataParallel(cuda_model, device_ids=alt_device_ids),
            )
            args_1, args_2 = cuda_args, cuda_args
        else:
            raise AssertionError("DataParallel compare mode type is not valid.")

        def data_parallel_test_assert(self):
            if target_layer:
                attr_method_1 = algorithm(
                    model_1, get_nested_attr(model_1, target_layer)
                )
                # cuda_model is used to obtain target_layer since DataParallel
                # adds additional wrapper.
                # model_2 is always either the CUDA model itself or DataParallel
                if alt_device_ids is None:
                    attr_method_2 = algorithm(
                        model_2, get_nested_attr(cuda_model, target_layer)
                    )
                else:
                    # LayerDeepLift and LayerDeepLiftShap do not take device ids
                    # as a parameter, since they must always have the DataParallel
                    # model object directly.
                    # NeuronDeconvolution and NeuronGuidedBackprop also require the
                    # model and cannot take a forward function.
                    if issubclass(
                        algorithm,
                        (
                            LayerDeepLift,
                            LayerDeepLiftShap,
                            NeuronDeepLift,
                            NeuronDeepLiftShap,
                            NeuronDeconvolution,
                            NeuronGuidedBackprop,
                            GuidedGradCam,
                        ),
                    ):
                        attr_method_2 = algorithm(
                            model_2, get_nested_attr(cuda_model, target_layer)
                        )
                    else:
                        attr_method_2 = algorithm(
                            model_2.forward,
                            get_nested_attr(cuda_model, target_layer),
                            device_ids=alt_device_ids,
                        )
            else:
                attr_method_1 = algorithm(model_1)
                attr_method_2 = algorithm(model_2)

            if noise_tunnel:
                attr_method_1 = NoiseTunnel(attr_method_1)
                attr_method_2 = NoiseTunnel(attr_method_2)
            if attr_method_1.has_convergence_delta():
                attributions_1, delta_1 = attr_method_1.attribute(
                    return_convergence_delta=True, **args_1
                )
                self.setUp()
                attributions_2, delta_2 = attr_method_2.attribute(
                    return_convergence_delta=True, **args_2
                )
                assertTensorTuplesAlmostEqual(
                    self, attributions_1, attributions_2, delta=target_delta, mode="max"
                )
                assertTensorTuplesAlmostEqual(
                    self, delta_1, delta_2, delta=target_delta, mode="max"
                )
            else:
                attributions_1 = attr_method_1.attribute(**args_1)
                self.setUp()
                attributions_2 = attr_method_2.attribute(**args_2)
                assertTensorTuplesAlmostEqual(
                    self, attributions_1, attributions_2, delta=target_delta, mode="max"
                )

        return data_parallel_test_assert


class DataParallelTest(BaseGPUTest, metaclass=DataParallelMeta):
    pass
