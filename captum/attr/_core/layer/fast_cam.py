#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from captum.log import log_usage
from captum.attr import LayerActivation


class LayerFastCam(LayerActivation):
    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        LayerActivation.__init__(self, forward_func, layer, device_ids)
        
    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        scale: str = "smole",
        norm: str = "gamma",
        scale_args: Dict[str, float] = None,
        norm_args: Dict[str, float] = None,
        relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        
        out = super().attribute(inputs,
                                additional_forward_args, 
                                attribute_to_layer_input)
        if scale == "smole":
            out = smole_scale(out)
            
        if norm == "gamma":
            out = gamma_norm(out)
        return out
        
    
def smole_scale(inputs):
    x = inputs + 1e-8
    m = x.mean(dim=1)
    k = torch.log2(m) - torch.mean(torch.log2(x), dim=1)
    th = k * m
    return th

def gamma_norm(inputs):       
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
        _iter    = _iter - 2
        gs      = _gamma(s)
        L       = x.pow(s) * gs * torch.exp(-x)
        gs      *= s                  # Gamma(s + 1)
        R       = torch.reciprocal(gs) * torch.ones_like(x)
        X       = x                   # x.pow(1)
        for k in range(_iter):
            gs      *= s + k + 1.0    # Gamma(s + k + 2)
            R       += X / gs 
            X       = X*x             # x.pow(k+1)
        gs      *= s + _iter + 1.0     # Gamma(s + iter + 2)
        R       += X / gs
        return  L * R
    
    def _trigamma(x):
        z   = x + 1.0
        zz  = z.pow(2.0)
        a   = 0.2 - torch.reciprocal(7.0*zz)
        b   = 1.0 - a/zz 
        c   = 1.0 + b/(3.0 * z)
        d   = 1.0 + c/(2.0 * z)
        e   = d/z 
        e   = e + torch.reciprocal(x.pow(2.0))
        return e

    def _k_update(k,s):
        nm = torch.log(k) - torch.digamma(k) - s
        dn = torch.reciprocal(k) - _trigamma(k)
        k2 = k - nm/dn
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
    eps = 1e-8
 
    s0, s1, s2 = inputs.size()
    x = inputs.reshape(s0, s1 * s2) 
    x = x - torch.min(x, dim=1)[0] + eps
    k, th = _compute_ml_est(x)
    x = (1.0 / _gamma(k)) * _lower_incl_gamma(k, x / th)
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    x = x.reshape(s0, s1, s2)
    return x  