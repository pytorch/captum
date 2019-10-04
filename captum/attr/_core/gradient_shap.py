#!/usr/bin/env python3
import torch

import numpy as np

from .._utils.attribution import GradientAttribution
from .._utils.common import _format_attributions

from .noise_tunnel import NoiseTunnel


class GradientShap(GradientAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (function): The forward function of the model or
                       any modification of it
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs,
        baselines,
        n_samples=50,
        stdevs=0.0,
        target=None,
        additional_forward_args=None,
    ):
        r"""
        Implements gradient SHAP based on the implementation from SHAP's primary
        author. For reference, please, view:

        https://github.com/slundberg/shap/#deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models

        A Unified Approach to Interpreting Model Predictions
        http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

        GradientShap approximates SHAP values by computing the expectations of
        gradients by randomly sampling from the distribution of baselines/references.
        It adds white noise to each input sample `n_samples` times, selects a
        random point along the path between baseline and input, and computes the
        gradient of outputs with respect to those selected random points.
        The final SHAP values represent the expected values of
        gradients * (inputs - baselines).

        GradientShap makes an assumption that the input features are independent
        and that there is a linear relationship between current inputs and the
        baselines/references. Under those assumptions, SHAP value can be
        approximated as the expectation of gradients that are computed for randomly
        generated `n_samples` input samples after adding gaussian noise `n_samples`
        times to each input for different baselines/references.

        In some sense it can be viewed as an approximation of integrated gradients
        by computing the expectations of gradients for different baselines.

        Current implementation uses Smoothgrad from `NoiseTunnel` in order to
        randomly draw samples from the distribution of baselines, add noise to input
        samples and compute the expectation (smoothgrad).

        Args:

            inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if mutliple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (tensor or tuple of tensors, optional):  Baselines define
                        the starting point from which expectation is computed.
                        If inputs is a single tensor, baselines must also be a
                        single tensor.
                        If inputs is a tuple of tensors, baselines must also be
                        a tuple of tensors, with the same number of tensors as
                        the inputs. The first dimension in baseline tensors
                        defines the distribution from which we randomly draw
                        samples. All other dimensions starting after
                        the first dimension should match with the inputs'
                        dimensions after the first dimension. It is recommended that
                        the number of samples in the baselines' tensors is larger
                        than one.
                        Default: zero tensor for each input tensor
            target (int, optional):  Output index for which gradient is computed
                        (for classification cases, this is the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary. (Note: Tuples for multi
                        -dimensional output indices will be supported soon.)
            additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It can contain a tuple of ND tensors or
                        any arbitrary python type of any shape.
                        In case of the ND tensor the first dimension of the
                        tensor must correspond to the batch size. It will be
                        repeated for each `n_steps` for each randomly generated
                        input sample.
                        Note that the gradients are not computed with respect
                        to these arguments.
                        Default: None
            n_samples (int, optional):  The number of randomly generated examples
                        per sample in the input batch. Random examples are
                        generated by adding gaussian random noise to each sample.
                        Default: `5` if `n_samples` is not provided.

        Returns:

            attributions (tensor or tuple of tensors): Attribution score
                        computed based on GradientSHAP with respect
                        to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            delta (float): This is computed using the property that the total
                        sum of forward_func(inputs) - forward_func(baselines)
                        must be very colse to the total sum of the attributions
                        based on GradientSHAP.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> net = ImageClassifier()
                >>> gradient_shap = GradientShap(net)
                >>> input = torch.randn(3, 3, 32, 32, requires_grad=True)
                >>> # choosing baselines randomly
                >>> baselines = torch.randn(20, 3, 32, 32)
                >>> # Computes gradient shap for the input
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution, delta = gradient_shap.attribute(input, baselines,
                                                                 target=5)

        """
        input_min_baseline_x_grad = InputBaselineXGradient(self.forward_func)

        nt = NoiseTunnel(input_min_baseline_x_grad)
        attributions, delta = nt.attribute(
            inputs,
            nt_type="smoothgrad",
            n_samples=n_samples,
            stdevs=stdevs,
            draw_baseline_from_distrib=True,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        delta = abs(torch.mean(delta.reshape(-1, n_samples), dim=1)).sum().item()
        return attributions, delta

    def _has_convergence_delta(self):
        return True


class InputBaselineXGradient(GradientAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (function): The forward function of the model or
                       any modification of it
        """
        super().__init__(forward_func)

    def attribute(
        self, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        def scale_input(input, baseline, rand_coefficient):
            # batch size
            bsz = input.shape[0]
            inp_shape_wo_bsz = input.shape[1:]
            inp_shape = (bsz,) + tuple([1] * len(inp_shape_wo_bsz))

            # expand and reshape the indices
            rand_coefficient = rand_coefficient.view(inp_shape).requires_grad_()

            input_baseline_scaled = (
                rand_coefficient * input + (1 - rand_coefficient) * baseline
            )
            return input_baseline_scaled

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        rand_coefficient = torch.tensor(
            np.random.uniform(0.0, 1.0, inputs[0].shape[0]),
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )

        input_baseline_scaled = tuple(
            scale_input(input, baseline, rand_coefficient)
            for input, baseline in zip(inputs, baselines)
        )
        grads = self.gradient_func(
            self.forward_func, input_baseline_scaled, target, additional_forward_args
        )

        input_baseline_diffs = tuple(
            input - baseline for input, baseline in zip(inputs, baselines)
        )
        attributions = tuple(
            input_baseline_diff * grad
            for input_baseline_diff, grad in zip(input_baseline_diffs, grads)
        )

        delta = self._compute_convergence_delta(
            attributions,
            baselines,
            inputs,
            additional_forward_args=additional_forward_args,
            target=target,
            delta_per_sample=True,
        )
        return _format_attributions(is_inputs_tuple, attributions), delta

    def _has_convergence_delta(self):
        return True
