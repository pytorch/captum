#!/usr/bin/env python3

from copy import deepcopy

import torch

from ..._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_baselines,
    _expand_and_update_target,
    _format_input,
    _format_tensor_into_tuples,
)
from .._utils.batching import _divide_and_aggregate_metrics


def default_perturb_func(inputs, perturb_radius=0.02):
    r"""A default function for generating perturbations of `inputs`
    within perturbation radius of `perturb_radius`.
    The users can override this function if they prefer to use a
    different perturbation function.
    """
    inputs = _format_input(inputs)
    perturbed_input = tuple(
        input
        + torch.FloatTensor(input.size(), device=input.device).uniform_(
            -perturb_radius, perturb_radius
        )
        for input in inputs
    )
    return perturbed_input


def sensitivity_max(
    explanation_func,
    inputs,
    perturb_func=default_perturb_func,
    perturb_radius=0.02,
    n_samples=10,
    norm_ord="fro",
    max_examples_per_batch=None,
    **kwargs,
):
    r"""
    Explanation sensitivity measures the extent of explanation change when
    the input is slightly perturbed. It has been shown that the models that
    have high explanation sensitivity are prone to adversarial attacks:
    `Interpretation of Neural Networks is Fragile`
    https://www.aaai.org/ojs/index.php/AAAI/article/view/4252

    `sensitivity_max` metric measures maximum sensitivity of an explanation
    using Monte Carlo sampling-based approximation. In order to do so it
    samples multiple data points from a sub-space of an L-Infinity
    ball that has a `perturb_radius` radius. The latter can be provided by
    users as an input argument. This functionality is provided by
    `default_perturb_func` function.
    If users prefer different sampling techniques they can provide
    `perturb_func` input argument that performs the sampling different way.

    Note that max sensitivity it similar to Lipschitz Continuity metric
    however it is more robust and easier to estimate.
    Since the explanation, that among others can be the attribution
    function, isn't always continues can lead to unbounded Lipschitz continuity.
    Therefore the latter isn't always appropriate.

    More about the Lipschitz Continuity Metric can also be found here
    `On the Robustness of Interpretability Methods`
    https://arxiv.org/pdf/1806.08049.pdf
    and
    `Towards Robust Interpretability with Self-Explaining Neural Networks`
    https://papers.nips.cc/paper\
    8003-towards-robust-interpretability-
    with-self-explaining-neural-networks.pdf

    More details about sensitivity max can be found here:
    `On the (In)fidelity and Sensitivity of Explanations`
    https://arxiv.org/pdf/1901.09392.pdf

    Args:

        explanation_func (callable):
                This function can be the `attribute` method of an
                attribution algorithm or any other explanation method
                that returns the explanations.

        inputs (tensor or tuple of tensors):  Input for which
                explanations are computed. If `explanation_func` takes a
                single tensor as input, a single input tensor should
                be provided.
                If `explanation_func` takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        perturb_func (callable):
                The perturbation function of model inputs. This function takes
                model inputs and optionally `perturb_radius` if
                `default_perturb_func` is used and returns
                perturbed inputs.

                If there are more than one inputs passed to sensitivity function those
                will be passed to `perturb_func` as tuples in the same order as they
                are passed to sensitivity function.

                It is important to note that for performance reasons `perturb_func`
                isn't called for each example individually but on a batch of
                input examples that are repeated `max_examples_per_batch / batch_size`
                times within the batch.

            Default: default_perturb_func
        perturb_radius (float, optional): The epsilon radius of L-Infinity
            ball that is used for uniform random sampling which ultimately
            serve as input perturbations.
            This argument is passed to `perturb_func` if it is set to default
            `default_perturb_func`.

            Default: 0.02
        n_samples (int, optional): The number of times input tensors are perturbed.
                Each input example in the inputs tensor is expanded `n_samples`
                times before calling `perturb_func` function.

                Default: 10
        norm_ord (str, optional): The type of norm that is used to compute the
                norm of the sensitivity matrix which is defined as the difference
                between the explanation function at its input and perturbed input.

                Default: "fro"
        max_examples_per_batch (int, optional): The number of maximum input
                examples that are processed together. In case the number of
                examples (`input batch size * n_samples`) exceeds
                `max_examples_per_batch`, they will be sliced
                into batches of `max_examples_per_batch` examples and processed
                in a sequential order. If `max_examples_per_batch` is None, all
                examples are processed together.

                Default: None
         **kwargs (Any, optional): Contains a list of arguments that are passed
                to `explanation_func` explanation function which in some cases
                could be the `attribute` function of an attribution algorithm.
                Any additional arguments that need be passed to the explanation
                function should be included here.
                For instance, such arguments include:
                `additional_forward_args`, `baselines` and `target`.

    Returns:

        sensitivities (tensor): A tensor of scalar sensitivity scores per
               input example. The first dimension is equal to the
               number of examples in the input batch and the second
               dimension is one.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes sensitivity score for saliency maps
        >>> sens = sensitivity_max(net, saliency.attribute)

    """

    def _generate_perturbations(current_n_samples):
        r"""
        The perturbations are generated for each example `current_n_samples` times.

        For perfomance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_samples` repeated instances per example.
        """
        inputs_expanded = tuple(
            torch.repeat_interleave(input, current_n_samples, dim=0) for input in inputs
        )
        if len(inputs_expanded) == 1:
            inputs_expanded = inputs_expanded[0]

        return (
            perturb_func(inputs_expanded, perturb_radius)
            if perturb_func == default_perturb_func
            else perturb_func(inputs_expanded)
        )

    def max_values(input_tnsr):
        return torch.max(input_tnsr, dim=1).values

    def _next_sensitivity_max(current_n_samples):
        expl_inputs = explanation_func(inputs, **kwargs)

        inputs_perturbed = _generate_perturbations(current_n_samples)

        # copy kwargs and update some of the arguments that need to be expanded
        kwargs_copy = deepcopy(kwargs)
        _expand_and_update_additional_forward_args(current_n_samples, kwargs_copy)
        _expand_and_update_target(current_n_samples, kwargs_copy)
        _expand_and_update_baselines(inputs, current_n_samples, kwargs_copy)

        expl_perturbed_inputs = explanation_func(inputs_perturbed, **kwargs_copy)

        # tuplize `expl_perturbed_inputs` in case it is not
        expl_perturbed_inputs = _format_tensor_into_tuples(expl_perturbed_inputs)

        expl_inputs_expanded = tuple(
            expl_input.repeat_interleave(current_n_samples, dim=0)
            for expl_input in expl_inputs
        )

        sensitivities = torch.cat(
            [
                (expl_input - expl_perturbed).view(expl_perturbed.size(0), -1)
                for expl_perturbed, expl_input in zip(
                    expl_perturbed_inputs, expl_inputs_expanded
                )
            ],
            dim=1,
        )
        # compute the norm of original input explanations
        expl_inputs_norm_expanded = torch.norm(
            torch.cat(
                [expl_input.view(expl_input.size(0), -1) for expl_input in expl_inputs],
                dim=1,
            ),
            p=norm_ord,
            dim=1,
            keepdim=True,
        ).repeat_interleave(current_n_samples, dim=0)
        expl_inputs_norm_expanded = torch.where(
            expl_inputs_norm_expanded == 0.0,
            torch.tensor(
                1.0,
                device=expl_inputs_norm_expanded.device,
                dtype=expl_inputs_norm_expanded.dtype,
            ),
            expl_inputs_norm_expanded,
        )

        # compute the norm for each input noisy example
        sensitivities_norm = (
            torch.norm(sensitivities, p=norm_ord, dim=1, keepdim=True)
            / expl_inputs_norm_expanded
        )
        return max_values(sensitivities_norm.view(bsz, -1))

    inputs = _format_input(inputs)

    bsz = inputs[0].size(0)

    with torch.no_grad():
        metrics_sum = _divide_and_aggregate_metrics(
            inputs,
            n_samples,
            _next_sensitivity_max,
            max_examples_per_batch=max_examples_per_batch,
            agg_func=torch.max,
        )
    return metrics_sum
