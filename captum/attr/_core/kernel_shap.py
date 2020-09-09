#!/usr/bin/env python3

from typing import Callable
import torch
from torch import Tensor

from .lime import Lime

import math


def linear_regression_interpretable_model_trainer(
    interp_inputs: Tensor, exp_outputs: Tensor, weights: Tensor, **kwargs
):
    try:
        from sklearn import linear_model
    except:
        raise AssertionError(
            "Requires sklearn for default interpretable model training with Lasso regression. Please install sklearn or use a custom interpretable model training function."
        )
    clf = linear_model.LinearRegression()
    clf.fit(
        interp_inputs.cpu().numpy(), exp_outputs.cpu().numpy(), weights.cpu().numpy()
    )
    return torch.from_numpy(clf.coef_)


def combination(n, k):
    try:
        # Combination only available in Python 3.8
        return math.comb(n, k)
    except:
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def kernel_shap_similarity_kernel(
    original_inp, sampled_inp, interpretable_sample, **kwargs
):
    assert (
        "num_interp_features" in kwargs
    ), "Must provide num_interp_features to use default similarity kernel"
    num_selected_features = int(interpretable_sample.sum(dim=1).item())
    num_features = kwargs["num_interp_features"]
    combinations = combination(num_features, num_selected_features)
    denom = (
        combinations * num_selected_features * (num_features - num_selected_features)
    )
    if denom != 0:
        similarities = (num_features - 1) / denom
    else:
        similarities = 100
    return torch.tensor([similarities])


class KernelShap(Lime):
    r"""
    Integrated Gradients is an axiomatic model interpretability algorithm that
    assigns an importance score to each input feature by approximating the
    integral of gradients of the model's output with respect to the inputs
    along the path (straight line) from given baselines / references to inputs.

    Baselines can be provided as input arguments to attribute method.
    To approximate the integral we can choose to use either a variant of
    Riemann sum or Gauss-Legendre quadrature rule.

    More details regarding LIME can be found in the
    original paper:
    https://arxiv.org/abs/1703.01365

    """

    def __init__(self, forward_func: Callable) -> None:
        Lime.__init__(
            self,
            forward_func,
            linear_regression_interpretable_model_trainer,
            kernel_shap_similarity_kernel,
        )
