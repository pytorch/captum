#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from captum.log import log_usage

from .._utils.attribution import GradientAttribution, LayerAttribution
from .layer.layer_activation import LayerActivation 
from ..._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric


class MultiscaleFastCam(GradientAttribution):
    def __init__(
        self,
        forward_func: Callable,
        layers: [Module, ...],
        scale: Any = "gamma",
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self.layer_acts = [LayerActivation(forward_func, l, device_ids) for l in layers]
        self.scale_func = self._set_scaling_func(scale)
    
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
    ) -> [Tensor, ...]:

        attributes = []
        for layer_act in self.layer_acts:
            layer_attr = layer_act.attribute(inputs,
                                             additional_forward_args,
                                             attribute_to_layer_input)
            smoe_attr = self._compute_smoe(layer_attr)
            scaled_attr = self.scale_func(smoe_attr)
            attributes.append(scaled_attr)
        return attributes
    
    @staticmethod
    def combine(
        saliency_maps: [Tensor, ...],
        weights: [float, ...],
        output_shape: Tuple, 
        resize_mode: str = "bilinear",
        relu_attribution: bool = False
    ) -> (Tensor, [Tensor, ...]):
        
        assert len(saliency_maps) > 1, "need more than 1 saliency map to combine."
        assert len(weights) == len(saliency_maps), "weights and saliency maps \
            should have the same length."

        bn  = saliency_maps[0].size()[0]
        combined_map  = torch.zeros((bn, 1, output_shape[0], output_shape[1]),
                          dtype=saliency_maps[0].dtype,
                          device=saliency_maps[0].device)
        weighted_map  = []
        for i, smap in enumerate(saliency_maps):
            w = F.interpolate(smap.unsqueeze(1),
                              size=output_shape,
                              mode=resize_mode,
                              align_corners=False)
            weighted_map.append(w)
            combined_map  += (w * weights[i])
        combined_map  = combined_map / np.sum(weights)
        combined_map  = combined_map.reshape(bn, output_shape[0], output_shape[1])
        weighted_map  = torch.stack(weighted_map, dim=1)
        weighted_map  = weighted_map.reshape(bn, len(weights), output_shape[0], output_shape[1])
        if relu_attribution:
            combined_map = F.relu(combined_map)
            weighted_map = F.relu(weighted_map)
        return combined_map, weighted_map

    def _set_scaling_func(self, scale):
        if scale == "gamma":
            return self._compute_gamma_scale
        elif scale == "normal":
            return self._compute_normal_scale
        elif scale is None or scale == "None":
            return lambda x: x
        elif callable(scale):
            return scale
        else:
            raise NameError(f"{scale} scaling option not found or invalid")

    def _compute_smoe(self, inputs):
        x = inputs + 1e-7
        m = x.mean(dim=1)
        k = torch.log2(m) - torch.mean(torch.log2(x), dim=1)
        th = k * m
        return th

    def _compute_gamma_scale(self, inputs):
        def _gamma(z):
            x = torch.ones_like(z) * 0.99999999999980993
            for i in range(8):
                i1  = torch.tensor(i + 1.0)
                x   = x + cheb[i] / (z + i1)
            t = z + 8.0 - 0.5
            y = two_pi * t.pow(z+0.5) * torch.exp(-t) * x
            y = y / z
            return y   
        
        def _lower_incl_gamma(s, x, _iter=8):
            _iter = _iter - 2
            gs = _gamma(s)
            L = x.pow(s) * gs * torch.exp(-x)
            gs *= s                     # Gamma(s + 1)
            R = torch.reciprocal(gs) * torch.ones_like(x)
            X = x                       # x.pow(1)
            for k in range(_iter):
                gs *= s + k + 1.0       # Gamma(s + k + 2)
                R += X / gs 
                X = X*x                 # x.pow(k+1)
            gs *= s + _iter + 1.0       # Gamma(s + iter + 2)
            R += X / gs
            return  L * R
        
        def _trigamma(x):
            z = x + 1.0
            zz = z.pow(2.0)
            a = 0.2 - torch.reciprocal(7.0 * zz)
            b = 1.0 - a / zz 
            c = 1.0 + b / (3.0 * z)
            d = 1.0 + c / (2.0 * z)
            e = d / z + torch.reciprocal(x.pow(2.0))
            return e

        def _k_update(k,s):
            nm = torch.log(k) - torch.digamma(k) - s
            dn = torch.reciprocal(k) - _trigamma(k)
            k2 = k - nm / dn
            return k2
                
        def _compute_ml_est(x, i=10):
            x  = x + eps
            s  = torch.log(torch.mean(x, dim=1)) - torch.mean(torch.log(x),dim=1)
            s3 = s - 3.0
            rt = torch.sqrt(s3.pow(2.0) + 24.0 * s)
            nm = 3.0 - s + rt
            dn = 12.0 * s
            k  = nm / dn + eps
            for _ in range(i):
                k =  _k_update(k,s)
            k   = torch.clamp(k, eps, 18.0)
            th  = torch.reciprocal(k) * torch.mean(x,dim=1)
            return k, th
        
        cheb = torch.tensor([676.5203681218851, -1259.1392167224028, 
                            771.32342877765313, -176.61502916214059, 
                            12.507343278686905, -0.13857109526572012,
                            9.9843695780195716e-6, 1.5056327351493116e-7])
        two_pi = torch.tensor(np.sqrt(2.0 * np.pi))
        eps = 1e-7
    
        s0, s1, s2 = inputs.size()
        x = inputs.reshape(s0, s1 * s2) 
        x = x - torch.min(x, dim=1)[0] + eps
        k, th = _compute_ml_est(x)
        x = (1.0 / _gamma(k)) * _lower_incl_gamma(k, x / th)
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        output = x.reshape(s0, s1, s2)
        return output

        