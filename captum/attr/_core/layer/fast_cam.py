#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from captum.log import log_usage

from ...._utils.common import _format_output
from ...._utils.gradient import _forward_layer_eval
from ..._utils.attribution import LayerAttribution


class LayerFastCam(LayerAttribution):
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
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's output
                          dimensions, except for dimension 2, which will be 1,
                          since GradCAM sums over channels.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        self.layer_act = LayerAttribution.__init__(self, forward_func, layer, device_ids)
        
    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        apply_gamma_norm: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the outputs of internal layers, depending on whether we
                        attribute to the input or output, are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            apply_gamma_norm (bool, optional): If set to true, apply gamma scale 
                        norm to saliency maps. 

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attributions based on GradCAM method.
                        Attributions will be the same size as the
                        output of the given layer, except for dimension 2,
                        which will be 1 due to summing over channels.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.
        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx50x8x8.
            >>> net = ImageClassifier()
            >>> attrs = [LayerFastCam(net, net.layer1[2]),
                         LayerFastCam(net, net.layer3[5]),
                         LayerFastCam(net, net.layer5[2])]
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> layer_attributions = [attr.attribute(input, apply_gamma_norm=True) \
                                        for attr in attrs]
            >>> combined_attribution = LayerFastCam.combine(layer_attributions,
                                                            weights=[1.0, 1.0, 1.0],
                                                            output_shape=(32, 32))
        """
        with torch.no_grad():
            layer_attrs, is_layer_tuple = _forward_layer_eval(
                self.forward_func,
                inputs,
                self.layer,
                additional_forward_args,
                device_ids=self.device_ids,
                attribute_to_layer_input=attribute_to_layer_input,
            )
        # return _format_output(is_layer_tuple, layer_attrs)
        # return layer_attrs
        scaled_attributes = []
        for layer_i, layer_attr in enumerate(layer_attrs):
            scaled_attribute = self._compute_smoe_scale(layer_attr)
            if apply_gamma_norm:
                scaled_attribute = self._compute_gamma_norm(scaled_attribute)
            scaled_attributes.append(scaled_attribute)
        return _format_output(is_layer_tuple, tuple(scaled_attributes))

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

    def _compute_smoe_scale(
        self,
        inputs: Tensor
        ) -> Tensor:
        r"""
        """
        x = inputs + 1e-7
        m = x.mean(dim=1)
        k = torch.log2(m) - torch.mean(torch.log2(x), dim=1)
        th = k * m
        return th

    def _compute_gamma_norm(
        self,
        inputs: Tensor
        ) -> Tensor:

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
