#!/usr/bin/env python3
import unittest
from typing import List

import torch
from torch.nn import Module

from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._core.multiscale_fast_cam import MultiscaleFastCam

from ..helpers.basic import BaseTest, assertTensorAlmostEqual
from ..helpers.basic_models import BasicModel_ConvNet_One_Conv


class Test(BaseTest):
    def test_simple_single_input(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        target = 0
        scale = 'smoe'
        norm = 'gaussian'
        ex = [
            [0.1859, 0.3698, 0.7376, 0.9216],
            [0.2559, 0.3854, 0.6444, 0.7739],
            [0.3960, 0.4167, 0.4580, 0.4786],
            [0.4661, 0.4323, 0.3647, 0.3310]
        ]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def test_simple_multi_inputs(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp = inp.repeat(2, 1, 1, 1)
        scale = 'smoe'
        norm = 'gaussian'
        ex = [
        [
            [0.1859, 0.3698, 0.7376, 0.9216],
            [0.2559, 0.3854, 0.6444, 0.7739],
            [0.3960, 0.4167, 0.4580, 0.4786],
            [0.4661, 0.4323, 0.3647, 0.3310]
        ],
        [
            [0.1859, 0.3698, 0.7376, 0.9216],
            [0.2559, 0.3854, 0.6444, 0.7739],
            [0.3960, 0.4167, 0.4580, 0.4786],
            [0.4661, 0.4323, 0.3647, 0.3310]
        ]]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def test_simple_single_input_gamma(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        scale = 'smoe'
        norm = 'gamma'
        ex = [
            [0.0338, 0.2465, 0.6719, 0.8846],
            [0.2170, 0.3724, 0.6833, 0.8387],
            [0.5834, 0.6243, 0.7061, 0.7470],
            [0.7666, 0.7502, 0.7175, 0.7011]
        ]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def test_simple_single_input_std(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        scale = 'std'
        norm = 'gaussian'
        ex = [
            [0.1565, 0.1771, 0.2182, 0.2388],
            [0.3186, 0.3340, 0.3649, 0.3804],
            [0.6429, 0.6480, 0.6583, 0.6634],
            [0.8050, 0.8050, 0.8050, 0.8050]
        ]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def test_simple_single_input_mean(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        scale = 'mean'
        norm = 'gaussian'
        ex = [
            [0.1825, 0.1932, 0.2145, 0.2252],
            [0.3089, 0.3289, 0.3687, 0.3886],
            [0.5618, 0.6002, 0.6771, 0.7155],
            [0.6882, 0.7359, 0.8313, 0.8790]
        ]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def test_simple_single_input_max(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        scale = 'max'
        norm = 'gaussian'
        ex = [
            [0.1713, 0.1854, 0.2135, 0.2276],
            [0.3116, 0.3300, 0.3670, 0.3855],
            [0.5921, 0.6194, 0.6739, 0.7011],
            [0.7324, 0.7640, 0.8273, 0.8590]
        ]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def test_simple_single_input_normal(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        layers = [net.relu1]
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        scale = 'normal'
        norm = 'gaussian'
        ex = [
            [0.0678, 0.2105, 0.4959, 0.6386],
            [0.2290, 0.3364, 0.5511, 0.6584],
            [0.5515, 0.5881, 0.6614, 0.6980],
            [0.7127, 0.7140, 0.7165, 0.7178]
        ]
        self._fastcam_test_assert(net, layers, inp, ex, scale, norm, ex)

    def _fastcam_test_assert(
        self,
        model: Module,
        layers: List[Module],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected, 
        scale: str,
        norm: str,
        combine: bool = False
    ) -> None:
        fastcam = MultiscaleFastCam(model, layers=layers)
        attributions = fastcam.attribute(test_input,
                                         scale=scale,
                                         norm=norm,
                                         weights=None,
                                         combine=combine)
        if isinstance(test_input, tuple):
            for i in range(len(test_input)):
                assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)
        else:
            assertTensorAlmostEqual(self, attributions, expected, delta=0.01)
        



if __name__ == "__main__":
    unittest.main()
