#!/usr/bin/env python3

import torch
import numpy as np

from .utils.attribution import Attribution
from .utils.common import validate_reg_type, format_input, _format_attributions, zeros
from .integrated_gradients import IntegratedGradients

SUPPORTED_ALGORITHMS_RETURNING_DELTA = [IntegratedGradients]


class NoiseTunnel(Attribution):
    r"""
    Adds random noise to remove noise. It aggregates attribution results
    accross all random samples per input.
    """

    def __init__(self, attribution_method):
        r"""
        attribution_method: Attribution
        """
        self.attribution_method = attribution_method
        self.is_delta_supported = (
            type(self.attribution_method) in SUPPORTED_ALGORITHMS_RETURNING_DELTA
        )
        super().__init__()

    # TODO parallelize this accross multiple samples.
    # TODO add smoothgrad squared to it - https://arxiv.org/pdf/1806.10758.pdf
    def attribute(
        self, inputs, reg_type="smoothgrad", n_samples=20, noise_frac=0.2, **kwargs
    ):
        r"""
        This method generates random samples by adding Gaussian noise to an
        input sample. It computed the expected value(smoothgrad) or
        variance(vargrad) accross all random samples for given input and
        applies `attribution_method` to each of those samples.

        This method currently does not support a batch of input samples.
        It will be supported soon

         Papers: https://arxiv.org/abs/1810.03292
                 https://arxiv.org/abs/1810.03307
                 https://arxiv.org/abs/1706.03825
         Demos from Google PAIR team: https://pair-code.github.io/saliency/
            Args

                inputs:     A single high dimensional input tensor or a tuple of them.
                reg_type:   Regualrization type. Supported types are
                            `smoothgrad` or `vargrad`
                n_samples:  The number of ramdomly generated samples
                noise_frac: The fraction of noise that is added to an input to generate
                            a new sample


            Return

                attributions:  Integrated gradients with respect to each input feature
                delta       :  The difference between the expected and approximated
                               values of the attributions. It only applies
                               to some methods such as integrated gradients.

        """

        def add_reg_to_input(input, max, min, shape):
            stdev = noise_frac * (max - min)
            # use torch.normal once it starts supporting sizes
            noise = torch.tensor(np.random.normal(0, stdev, shape), dtype=input.dtype)
            return input + noise

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)

        validate_reg_type(reg_type)

        expected_attributions = zeros(inputs)
        expected_attributions_sq = zeros(inputs)

        # smoothgrad_Attr(x) = 1 / n * sum(Attr(x + N(0, sigma^2))
        inputs_max = tuple(torch.max(input).detach().item() for input in inputs)
        inputs_min = tuple(torch.min(input).detach().item() for input in inputs)
        inputs_shape = tuple(tuple(input.shape) for input in inputs)
        delta_accum = 0
        # TODO avoid this for-loop: try to override it with some optimizations
        # TODO we can get rid of this loop and simply use batching for attribution
        # TODO put random samples in the batch and run attribution only
        # once on that batch
        for _ in range(n_samples):
            inputs_noisy = tuple(
                add_reg_to_input(input, input_max, input_min, input_shape)
                for input_max, input_min, input_shape, input in zip(
                    inputs_max, inputs_min, inputs_shape, inputs
                )
            )
            attributions = self.attribution_method.attribute(inputs_noisy, **kwargs)
            if self.is_delta_supported:
                attributions, delta = attributions
                delta_accum += delta

            attributions = tuple(
                attribution.squeeze() for i, attribution in enumerate(attributions)
            )
            expected_attributions = tuple(
                expected_attributions[i] + attribution
                for i, attribution in enumerate(attributions)
            )
            expected_attributions_sq = tuple(
                expected_attributions_sq[i] + attribution * attribution
                for i, attribution in enumerate(attributions)
            )

        expected_attributions = tuple(
            expected_attribution * 1 / n_samples
            for i, expected_attribution in enumerate(expected_attributions)
        )
        delta_accum /= n_samples

        # TODO create an enum for this
        if reg_type == "smoothgrad":
            return self._apply_checks_and_return_attributions(
                expected_attributions, is_inputs_tuple, delta_accum
            )

        vargrad = tuple(
            expected_attribution_sq * 1 / n_samples
            - expected_attributions[i] * expected_attributions[i]
            for i, expected_attribution_sq in enumerate(expected_attributions_sq)
        )

        return self._apply_checks_and_return_attributions(
            vargrad, is_inputs_tuple, delta_accum
        )

    def _apply_checks_and_return_attributions(
        self, attributions, is_inputs_tuple, delta
    ):
        attributions = _format_attributions(is_inputs_tuple, attributions)

        return (attributions, delta) if self.is_delta_supported else attributions
