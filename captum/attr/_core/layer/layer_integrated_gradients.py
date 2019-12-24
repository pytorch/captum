#!/usr/bin/env python3
import torch

from torch.nn.parallel.scatter_gather import scatter

from captum.attr._utils.common import (
    _tensorize_baseline,
    _validate_input,
    _format_additional_forward_args,
    _format_attributions,
    _format_input_baseline,
)

from captum.attr._utils.gradient import _forward_layer_eval

from captum.attr._utils.attribution import LayerAttribution
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.gradient import _run_forward


class LayerIntegratedGradients(LayerAttribution, IntegratedGradients):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:
            forward_func (callable):  The forward function of the model or any
                          modification of it
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        IntegratedGradients.__init__(self, forward_func)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        internal_batch_size=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False,
    ):
        inps, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inps, baselines, n_steps, method)

        baselines = _tensorize_baseline(inps, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)
        inputs_layer = _forward_layer_eval(
            self.forward_func,
            inps,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        baselines_layer = _forward_layer_eval(
            self.forward_func,
            baselines,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        # inputs -> these inputs are scaled
        def gradient_func(
            forward_fn, inputs, target_ind=None, additional_forward_args=None
        ):
            if self.device_ids is None:
                scattered_inputs = inputs
            else:
                scattered_inputs = (
                    scatter_inp[0]
                    for scatter_inp in scatter(inputs, target_gpus=self.device_ids)
                )

            scattered_inputs_dict = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }

            with torch.autograd.set_grad_enabled(True):

                def layer_forward_hook(module, hook_inputs, hook_outputs=None):
                    return scattered_inputs_dict[hook_inputs[0].device]

                if attribute_to_layer_input:
                    hook = self.layer.register_forward_pre_hook(layer_forward_hook)
                else:
                    hook = self.layer.register_forward_hook(layer_forward_hook)

                output = _run_forward(
                    self.forward_func, additional_forward_args, target_ind,
                )
                hook.remove()
                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(torch.unbind(output), inputs)
            return grads

        self.gradient_func = gradient_func
        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )
        attributions = IntegratedGradients.attribute(
            self,
            inputs_layer,
            baselines=baselines_layer,
            target=target,
            additional_forward_args=all_inputs,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=False,
        )

        # TODO this needs to be formated properly -
        # it assumes that layer returns a tensor
        attributions = (attributions,)

        if return_convergence_delta:
            start_point, end_point = baselines, inps
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            # TODO this needs to be checked of tensor properly: len(attributions) > 1
            return _format_attributions(len(attributions) > 1, attributions), delta
        return _format_attributions(len(attributions) > 1, attributions)

    def has_convergence_delta(self):
        return True
