#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from captum.log import log_usage

from ..._utils.typing import ModuleOrModuleList
from .._utils.attribution import GradientAttribution
from .layer.layer_activation import LayerActivation


class MultiscaleFastCam(GradientAttribution):
    r"""
    Compute saliency map using Saliency Map Order Equivalence (SMOE). This
    method first computes the layer activation, then passes activations through
    a nonlinear SMOE function.

    The recommended use case for FastCAM is to compute saliency maps for multiple
    layers with different scales in a deep network, then combine them to obtain
    a more meaningful saliency map for the original input.

    More details regrading FastCam can be found in the original paper:
    https://arxiv.org/abs/1911.11293
    """

    def __init__(
        self,
        forward_func: Callable,
        layers: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:
            forward_func (callable): The forward function of the model or any
                          modification of it
            layers (torch.nn.Module or listt(torch.nn.Module)): A list of layers
                          for which attributions.
                          are computed.
            norm (str): choice of norm to normalize saliency maps
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        GradientAttribution.__init__(self, forward_func)
        self.layer_act = LayerActivation(forward_func, layers, device_ids)

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        scale: str = 'smoe',
        norm: str = 'gaussian',
        weights: List[float] = None,
        return_weighted=True,
        resize_mode: str = "bilinear",
        relu_attribution: bool = False,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False
    ) -> Tuple[Tensor, ...]:  

        # pick out functions
        self.scale_func = self.pick_scale_func(scale)
        self.norm_func = self.pick_norm_func(norm)
        if not weights:
            weights = np.ones(len(self.layers))

        layer_attrs = self.layer_act.attribute(
            inputs, additional_forward_args, attribute_to_layer_input
        )
        attributes = []
        for layer_attr in layer_attrs:
            scaled_attr = self.scale_func(layer_attr)
            normed_attr = self.norm_func(scaled_attr)
            attributes.append(normed_attr)
        attributes = tuple(attributes)

        ## Combine
        bn, channels, height, width = inputs.shape
        combined_map = torch.zeros(
            (bn, 1, height, width),
            dtype=attributes[0].dtype,
            device=attributes[0].device,
        )
        weighted_maps = [[] for _ in range(bn)]  # type: List[List[Any]]
        for m, smap in enumerate(attributes):
            for i in range(bn):
                w = F.interpolate(
                    smap[i].unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode=resize_mode,
                    align_corners=False,
                ).squeeze()
                weighted_maps[i].append(w)
                combined_map[i] += w * weights[m]
        combined_map = combined_map / np.sum(weights)
        weighted_maps = torch.stack([torch.stack(wmaps) for wmaps in weighted_maps])

        if relu_attribution:
            combined_map = F.relu(combined_map)
            weighted_maps = F.relu(weighted_maps)

        if return_weighted:
            return weighted_maps
        return combined_map


    def pick_norm_func(self, norm):
        norm = norm.lower()
        if norm == 'gamma':
            norm_func = self._compute_gamma_norm
        elif norm == 'gaussian':
            norm_func = self._compute_gaussian_norm
        elif norm is None or norm == 'None':
            norm_func = lambda x: x.squeeze()
        elif callable(norm):
            norm_func = norm
        else:
            msg = (
                f"{norm} norming option not found or invalid. "
                + "Available options: [gamma, normal, None]"
            )
            raise NameError(msg)
        return norm_func


    def pick_scale_func(self, scale):
        scale = scale.lower()
        if scale == 'smoe':
            scale_func = self._compute_smoe_scale
        elif scale == 'std':
            scale_func = self._compute_std_scale
        elif scale == 'mean':
            scale_func = self._compute_mean_scale
        elif scale == 'max':
            scale_func = self._compute_max_scale
        elif scale == 'normal':
            scale_func = _compute_normal_entropy_scale
        else:
            msg = (
                f"{scale} scaling option not found or invalid. "
                + "Available options: [smoe, std, mean, normal]"
            )
            raise NameError(msg)
        return scale_func

    def _compute_smoe_scale(self, inputs):
        x = inputs + 1e-7
        m = x.mean(dim=1, keepdims=True)
        k = torch.log2(m) - torch.log2(x).mean(dim=1, keepdims=True)
        th = k * m
        return th


    def _compute_std_scale(self, inputs):
        return torch.std(inputs, dim=1, keepdim=True)
    

    def _compute_mean_scale(self, inputs):
        return torch.mean(inputs, dim=1, keepdim=True)


    def _compute_max_scale(self, inputs):
        return torch.max(inputs, dim=1, keepdim=True)


    def _compute_normal_entropy_scale(self, inputs):
        c1 = torch.tensor(0.3989422804014327)  # 1.0/math.sqrt(2.0*math.pi)
        c2 = torch.tensor(1.4142135623730951)  # math.sqrt(2.0)
        c3 = torch.tensor(4.1327313541224930)
        def _compute_alpha(mean, std, a=0):
            return (a - mean) / std

        def _compute_pdf(eta):
            return c1 * torch.exp(-0.5 * eta.pow(2.0))

        def _compute_cdf(eta):
            e = torch.erf(eta / c2)
            return 0.5 * (1.0 + e) + 1e-7
        m = torch.mean(inputs, dim=1)
        s = torch.std(inputs, dim=1)
        a = _compute_alpha(m, s)
        pdf = _compute_pdf(a)  
        cdf = _compute_cdf(a)
        Z = 1.0 - cdf 
        T1 = torch.log(c3 * s * Z)
        T2 = (a * pdf) / (2.0 * Z)
        ent = T1 + T2
        return ent


    def _compute_gaussian_norm(self, inputs):
        b, _, h, w = inputs.size()
        x = inputs.reshape(b, h * w)
        m = x.mean(dim=1, keepdims=True)
        s = x.std(dim=1, keepdims=True)
        x = 0.5 * (1.0 + torch.erf((x - m) / (s * torch.sqrt(torch.tensor(2.0)))))
        x = x.reshape(b, h, w)
        return x


    def _compute_gamma_norm(self, inputs):
        def _gamma(z):
            x = torch.ones_like(z) * 0.99999999999980993
            for i in range(8):
                i1 = torch.tensor(i + 1.0)
                x = x + cheb[i] / (z + i1)
            t = z + 8.0 - 0.5
            y = two_pi * t.pow(z + 0.5) * torch.exp(-t) * x
            y = y / z
            return y

        def _lower_incl_gamma(s, x, _iter=8):
            _iter = _iter - 2
            gs = _gamma(s)
            L = x.pow(s) * gs * torch.exp(-x)
            gs *= s  # Gamma(s + 1)
            R = torch.reciprocal(gs) * torch.ones_like(x)
            X = x  # x.pow(1)
            for k in range(_iter):
                gs *= s + k + 1.0  # Gamma(s + k + 2)
                R += X / gs
                X = X * x  # x.pow(k+1)
            gs *= s + _iter + 1.0  # Gamma(s + iter + 2)
            R += X / gs
            return L * R

        def _trigamma(x):
            z = x + 1.0
            zz = z.pow(2.0)
            a = 0.2 - torch.reciprocal(7.0 * zz)
            b = 1.0 - a / zz
            c = 1.0 + b / (3.0 * z)
            d = 1.0 + c / (2.0 * z)
            e = d / z + torch.reciprocal(x.pow(2.0))
            return e

        def _k_update(k, s):
            nm = torch.log(k) - torch.digamma(k) - s
            dn = torch.reciprocal(k) - _trigamma(k)
            k2 = k - nm / dn
            return k2

        def _compute_ml_est(x, i=10):
            x = x + eps
            s = torch.log(x.mean(dim=1, keepdims=True)) 
            s = s - torch.log(x).mean(dim=1, keepdims=True)
            s3 = s - 3.0
            rt = torch.sqrt(s3.pow(2.0) + 24.0 * s)
            nm = 3.0 - s + rt
            dn = 12.0 * s
            k = nm / dn + eps
            for _ in range(i):
                k = _k_update(k, s)
            k = torch.clamp(k, eps, 18.0)
            th = torch.reciprocal(k) * torch.mean(x, dim=1, keepdims=True)
            return k, th

        cheb = torch.tensor(
            [
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ]
        )
        eps = 1e-7
        two_pi = torch.tensor(np.sqrt(2.0 * np.pi))
        b, _, h, w = inputs.size()
        x = inputs.reshape(b, h * w)
        x = x - torch.min(x, dim=1, keepdims=True)[0] + 1e-7
        k, th = _compute_ml_est(x)
        x = (1.0 / _gamma(k)) * _lower_incl_gamma(k, x / th)
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        output = x.reshape(b, h, w)
        return output
