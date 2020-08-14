#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from captum.log import log_usage

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
        layers: [Module, ...],
        norm: Any = "gamma",
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layers (list(torch.nn.Module)): A list of layers for which attributions
                          are computed.
            scale (str): choice of scale to normalize saliency maps
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        GradientAttribution.__init__(self, forward_func)
        self.layer_acts = [
            LayerActivation(forward_func, layer, device_ids) for layer in layers
        ]
        self.norm_func = self._set_norm_func(norm)

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
    ) -> [Tensor, ...]:
        r"""
        Args:

            inputs (tensor):  Input for which attributions
                        are computed. Input should have dimensions BHWC.
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
            list of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attributions based on FastCAM method.
                        Each element of the list is attributions computed from
                        layer 0.
        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx50x8x8.
            >>> input = torch.randn(1, 3, 32, 32)
            >>> fastcam = MultiscaleFastCam(model,
                                            layers=[model.relu,
                                                    model.layer1[2].relu,
                                                    model.layer2[3].relu,
                                                    model.layer3[5].relu,
                                                    model.layer4[2].relu],
                                            norm="gamma")
            >>> attributes = fastcam.attribute(transformed_img)
            >>> combined_map, weighted_maps = fastcam.combine(attributes,
                                weights=[1.0 for _ in range(5)],
                                output_shape=(in_height, in_width))
        """
        attributes = []
        for layer_act in self.layer_acts:
            layer_attr = layer_act.attribute(
                inputs, additional_forward_args, attribute_to_layer_input
            )
            smoe_attr = self._compute_smoe_scale(layer_attr)
            scaled_attr = self.norm_func(smoe_attr)
            attributes.append(scaled_attr)
        return tuple(attributes)

    @staticmethod
    def combine(
        saliency_maps: [Tensor, ...],
        weights: [float, ...],
        output_shape: Tuple,
        resize_mode: str = "bilinear",
        relu_attribution: bool = False,
    ) -> (Tensor, [Tensor, ...]):
        """Combine multi-scale saliency map (attributions) by taking in saliency
                            maps from multiple layers of the network.
        Args:
            saliency_maps (list(Tensors)): A List of attributions with different
                            size. The direct way to use this is to pass in the
                            outputs of `attribute` function above.
            weights (list(float)): Weights for each saliency map
            output_shape (tuple): Specifies the output shape of saliency. In most
                            cases, this should be the same as input image shape.
            resize_mode (str): Resize mode for interpolation.
            relu_attribution (bool): Apply relu to saliency maps before returning
                            output.
        Returns:
            - **combined_map** (*tensor*): The combined and weighted saliency map
            - **weighted_maps** (list of *tensors*): A List of weighted maps,
                            interpolated to `output_shape`.
        """

        assert len(saliency_maps) > 1, "need more than 1 saliency map to combine."
        assert len(weights) == len(
            saliency_maps
        ), "weights and saliency maps \
            should have the same length."

        bn = saliency_maps[0].size()[0]
        combined_map = torch.zeros(
            (bn, 1, output_shape[0], output_shape[1]),
            dtype=saliency_maps[0].dtype,
            device=saliency_maps[0].device,
        )
        weighted_maps = [[] for _ in range(bn)]
        for m, smap in enumerate(saliency_maps):
            for i in range(bn):
                w = F.interpolate(
                    smap[i].unsqueeze(0).unsqueeze(0),
                    size=output_shape,
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
        return combined_map, weighted_maps

    def _set_norm_func(self, scale):
        if scale == "gamma":
            return self._compute_gamma_norm
        elif scale == "gaussian":
            return self._compute_gaussian_norm
        elif scale is None or scale == "None":
            return lambda x: x.squeeze()
        elif callable(scale):
            return scale
        else:
            msg = (
                f"{scale} scaling option not found or invalid. "
                + "Available options: [gamma, normal, None]"
            )
            raise NameError(msg)

    def _compute_smoe_scale(self, inputs):
        x = inputs + 1e-7
        m = x.mean(1, keepdims=True)
        k = torch.log2(m) - torch.log2(x).mean(dim=1, keepdims=True)
        th = k * m
        return th

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
            s = torch.log(x.mean(dim=1, keepdims=True)) \
                    - torch.log(x).mean(dim=1, keepdims=True)
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
        two_pi = torch.tensor(np.sqrt(2.0 * np.pi))
        eps = 1e-7

        b, _, h, w = inputs.size()
        x = inputs.reshape(b, h*w)
        x = x - torch.min(x, dim=1, keepdims=True)[0] + eps
        k, th = _compute_ml_est(x)
        x = (1.0 / _gamma(k)) * _lower_incl_gamma(k, x / th)
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        output = x.reshape(b, h, w)
        return output

    def _compute_gaussian_norm(self, inputs):
        b, _, h, w = inputs.size()
        x = inputs.reshape(b, h * w)
        m = x.mean(dim=1, keepdims=True)
        s = x.std(dim=1, keepdims=True)
        x = 0.5 * (1.0 + torch.erf((x - m) / (s * torch.sqrt(torch.tensor(2.0)))))
        x = x.reshape(b, h, w)
        return x
