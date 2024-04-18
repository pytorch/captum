#!/usr/bin/env python3
from typing import Any, Tuple, Union

import torch
from captum._utils.common import (
    _extract_device,
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _run_forward,
)

from captum._utils.gradient import _forward_layer_eval

from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._utils.attribution import LayerAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn.parallel.scatter_gather import scatter


class LayerFeaturePermutation(LayerAttribution, FeaturePermutation):
    r"""
    A perturbation based approach to computing layer attribution similar to
    LayerFeatureAblation, but using FeaturePermutation under the hood instead
    of FeatureAblation.
    """

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        perturbations_per_eval: int = 1,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        def layer_forward_func(*args) -> Tensor:
            layer_length = args[-1]
            layer_input = args[:layer_length]
            original_inputs = args[layer_length:-1]

            device_ids = self.device_ids
            if device_ids is None:
                device_ids = getattr(self.forward_func, "device_ids", None)

            all_layer_inputs = {}
            if device_ids is not None:
                scattered_layer_input = scatter(layer_input, target_gpus=device_ids)
                for device_tensors in scattered_layer_input:
                    all_layer_inputs[device_tensors[0].device] = device_tensors
            else:
                all_layer_inputs[layer_input[0].device] = layer_input

            def forward_hook(module, inp, out=None):
                device = _extract_device(module, inp, out)
                is_layer_tuple = (
                    isinstance(out, tuple)
                    if out is not None
                    else isinstance(inp, tuple)
                )
                if device not in all_layer_inputs:
                    raise AssertionError(
                        "Layer input not placed on appropriate "
                        "device. If using a DataParallel model, either provide the "
                        "DataParallel model as forward_func or provide device ids"
                        " to the constructor."
                    )
                if not is_layer_tuple:
                    return all_layer_inputs[device][0]
                return all_layer_inputs[device]

            hook = None
            try:
                hook = self.layer.register_forward_hook(forward_hook)
                eval = _run_forward(self.forward_func, original_inputs, target=target)
            finally:
                if hook is not None:
                    hook.remove()
            return eval

        with torch.no_grad():
            inputs = _format_tensor_into_tuples(inputs)
            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )
            layer_eval = _forward_layer_eval(
                self.forward_func,
                inputs,
                self.layer,
                additional_forward_args,
                device_ids=self.device_ids,
            )
            layer_eval_len = (len(layer_eval),)
            all_inputs = (
                (inputs + additional_forward_args + layer_eval_len)
                if additional_forward_args is not None
                else inputs + layer_eval_len
            )

            permutator = self.attributor(forward_func=layer_forward_func)

            layer_attribs = permutator.attribute.__wrapped__(
                permutator,
                inputs=inputs,
                target=target,
                additional_forward_args=all_inputs,
                feature_mask=feature_mask,
                perturbations_per_eval=perturbations_per_eval,
            )
            _attr = _format_output(len(layer_attribs) > 1, layer_attribs)

        return _attr

    @property
    def attributor(self):
        return FeaturePermutation
