#!/usr/bin/env python3
import typing
from typing import Any, Callable, List, Tuple, Union, Optional

import torch
from torch import Tensor

from captum.log import log_usage

from ..._utils.common import (
    _format_input,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from ..._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from .._utils.approximation_methods import approximation_parameters
from .._utils.attribution import PerturbationAttribution
from .._utils.batching import _batch_attribution
from .._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
    _construct_default_feature_mask,
)


class LimeBase(PerturbationAttribution):
    r"""
    Lime is an interpretability method that trains an interpretable surrogate model
    by sampling points around a specified input example and using model evaluations
    at these points to train a simpler interpretable 'surrogate' model, such as a
    linear model.

    LimeBase provides a generic framework to train a surrogate interpretable model. This
    differs from most other attribution methods, since the method returns a representation
    of the interpretable model (e.g. coefficients of the linear model). For a similar
    interface to other perturbation-based attribution methods, please use the Lime child
    class, which defines specific transformations for the interpretable model.

    LimeBase allows sampling points in either the interpretable space or the original input
    space to train the surrogate model. The interpretable space is a feature vector used to
    train the surrogate interpretable model; this feature space is often of smaller dimensionality
    than the original feature space in order for the surrogate model to be more interpretable.

    If sampling in the interpretable space, a transformation function must be provided to
    define how a vector sampled in the interpretable space can be transformed into an
    example in the original input space. If sampling in the original input space, a
    transformation function must be provided to define how the input can be transformed
    into its interpretable vector representation.


    """

    def __init__(
        self,
        forward_func: Callable,
        train_interpretable_model_func: Callable,
        similarity_func: Callable,
        sampling_func: Callable,
        sample_interpretable_space: bool,
        from_interp_rep_transform: Callable,
        to_interp_rep_transform: Callable,
    ) -> None:
        PerturbationAttribution.__init__(self, forward_func)
        self.train_interpretable_model_func = train_interpretable_model_func
        self.similarity_func = similarity_func
        self.sampling_func = sampling_func
        self.sample_interpretable_space = sample_interpretable_space
        self.from_interp_rep_transform = from_interp_rep_transform
        self.to_interp_rep_transform = to_interp_rep_transform

        if self.sample_interpretable_space:
            assert (
                self.from_interp_rep_transform is not None
            ), "Must provide transform from interpretable space to original input space when sampling from interpretable space."
        else:
            assert (
                self.to_interp_rep_transform is not None
            ), "Must provide transform from original input space to interpretable space."

    # The following overloaded method signatures correspond to the case where
    # return_convergence_delta is False, then only attributions are returned,
    # and when return_convergence_delta is True, the return type is
    # a tuple with both attributions and deltas.

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        **kwargs
    ) -> TensorOrTupleOfTensorsGeneric:

        bsz = inputs.shape[0] if isinstance(inputs, Tensor) else inputs[0].shape[0]
        expand_inputs = False
        with torch.no_grad():
            interpretable_inps = []
            similarities = []
            outputs = []

            curr_inputs = []
            expanded_additional_args = None
            expanded_target = None
            for i in range(n_samples):
                curr_sample = self.sampling_func(inputs, **kwargs)
                if self.sample_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    curr_inputs.append(
                        self.from_interp_rep_transform(curr_sample, inputs, **kwargs)
                    )
                    similarities.append(
                        self.similarity_func(
                            inputs, curr_inputs[-1], curr_sample, **kwargs
                        )
                    )
                else:
                    curr_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(curr_sample, inputs, **kwargs)
                    )
                    similarities.append(
                        self.similarity_func(
                            inputs, curr_sample, interpretable_inps[-1], **kwargs
                        )
                    )

                if (
                    perturbations_per_eval is not None
                    and len(curr_inputs) == perturbations_per_eval
                ):
                    if expanded_additional_args is None:
                        expanded_additional_args = (
                            _expand_additional_forward_args(
                                additional_forward_args, len(curr_inputs)
                            )
                            if additional_forward_args is not None
                            else None
                        )
                    if expanded_target is None:
                        expanded_target = _expand_target(target, len(curr_inputs))
                    if perturbations_per_eval == 1:
                        combined_inputs = curr_inputs[0]
                    else:
                        combined_inputs = _reduce_list(curr_inputs)
                    model_out = _run_forward(
                        self.forward_func,
                        combined_inputs,
                        expanded_target,
                        expanded_additional_args,
                    )
                    if (
                        not expand_inputs
                        and isinstance(model_out, Tensor)
                        and model_out.numel() != len(curr_inputs)
                    ):
                        assert model_out.numel() == bsz * len(
                            curr_inputs
                        ), "Number of outputs is not appropriate, must return one output per example. If forward function returns a scalar per batch, ensure that perturbations_per_eval is set to 1."
                        expand_inputs = True
                    outputs.append(
                        model_out
                        if isinstance(model_out, Tensor)
                        else torch.tensor(model_out)
                    )

                    curr_inputs = []

            if len(curr_inputs) > 0:
                expanded_additional_args = (
                    _expand_additional_forward_args(
                        additional_forward_args, len(curr_inputs)
                    )
                    if additional_forward_args is not None
                    else None
                )
                expanded_target = _expand_target(target, len(curr_inputs))
                model_out = _run_forward(
                    self.forward_func,
                    _reduce_list(curr_inputs),
                    expanded_target,
                    expanded_additional_args,
                )
                outputs.append(
                    model_out
                    if isinstance(model_out, Tensor)
                    else torch.tensor(model_out)
                )

                if (
                    not expand_inputs
                    and isinstance(model_out, Tensor)
                    and model_out.numel() != len(curr_inputs)
                ):
                    assert model_out.numel() == bsz * len(
                        curr_inputs
                    ), "Number of outputs is not appropriate, must return one output per example. If forward function returns a scalar per batch, ensure that perturbations_per_eval is set to 1."
                    expand_inputs = True

            combined_interp_inps = torch.cat(interpretable_inps)
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            )
            combined_sim = torch.cat(similarities)

            if expand_inputs:
                combined_interp_inps = torch.repeat_interleave(
                    combined_interp_inps, bsz, 0
                )
                combined_sim = torch.repeat_interleave(combined_sim, bsz, 0)

            interp_model = self.train_interpretable_model_func(
                combined_interp_inps, combined_outputs, combined_sim, **kwargs
            )
            return interp_model

    def has_convergence_delta(self) -> bool:
        return False

    @property
    def multiplies_by_inputs(self):
        return False


def lasso_interpretable_model_trainer(
    interp_inputs: Tensor, exp_outputs: Tensor, weights: Tensor, **kwargs
):
    try:
        from sklearn import linear_model
    except:
        raise AssertionError(
            "Requires sklearn for default interpretable model training with Lasso regression. Please install sklearn or use a custom interpretable model training function."
        )
    # print(interp_inputs)
    # print(exp_outputs)
    # print(weights)
    clf = linear_model.Lasso(alpha=kwargs["alpha"] if "alpha" in kwargs else 1.0)
    clf.fit(
        interp_inputs.cpu().numpy(), exp_outputs.cpu().numpy(), weights.cpu().numpy()
    )
    # print(clf.coef_)
    return torch.from_numpy(clf.coef_)


def default_from_interp_rep_transform(curr_sample, original_inputs, **kwargs):
    assert (
        "feature_mask" in kwargs
    ), "Must provide feature_mask to use default interpretable representation transform"
    assert (
        "baselines" in kwargs
    ), "Must provide baselines to use default interpretable representation transfrom"
    feature_mask = kwargs["feature_mask"]
    if isinstance(feature_mask, Tensor):
        binary_mask = curr_sample[0][feature_mask]
        return binary_mask * original_inputs + (1 - binary_mask) * kwargs["baselines"]
    else:
        binary_mask = tuple(
            curr_sample[0][feature_mask[j]] for j in range(len(feature_mask))
        )
        return tuple(
            binary_mask[j] * original_inputs[j]
            + (1 - binary_mask[j]) * kwargs["baselines"][j]
            for j in range(len(feature_mask))
        )


def default_similarity_kernel(original_inp, _, __, **kwargs):
    return torch.ones(1)


def default_sampling_func(original_inp, **kwargs):
    assert (
        "num_interp_features" in kwargs
    ), "Must provide num_interp_features to use default interpretable sampling function"
    if isinstance(original_inp, Tensor):
        # bsz = original_inp.shape[0]
        device = original_inp.device
    else:
        # bsz = original_inp[0].shape[0]
        device = original_inp[0].device

    probs = torch.ones(1, kwargs["num_interp_features"]) * 0.5
    return torch.bernoulli(probs).to(device=device)


class Lime(LimeBase):
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

    def __init__(
        self,
        forward_func: Callable,
        train_interpretable_model_func: Callable = lasso_interpretable_model_trainer,
        similarity_func: Callable = default_similarity_kernel,
    ) -> None:
        LimeBase.__init__(
            self,
            forward_func,
            train_interpretable_model_func,
            similarity_func,
            default_sampling_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

    # The following overloaded method signatures correspond to the case where
    # return_convergence_delta is False, then only attributions are returned,
    # and when return_convergence_delta is True, the return type is
    # a tuple with both attributions and deltas.

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = isinstance(inputs, tuple)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)

        if feature_mask is None:
            feature_mask, num_interp_features = _construct_default_feature_mask(
                formatted_inputs
            )
        else:
            feature_mask = _format_input(feature_mask)
            num_interp_features = (
                max(torch.max(single_inp).item() for single_inp in feature_mask) + 1
            )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable features. "
            )

        coefs = super().attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
        )
        if return_input_shape:
            # print(coefs)
            attr = [torch.zeros_like(inp) for inp in formatted_inputs]
            for tensor_ind in range(len(formatted_inputs)):
                for single_feature in range(num_interp_features):
                    attr[tensor_ind] += coefs[single_feature] * (
                        feature_mask[tensor_ind] == single_feature
                    )
            return _format_output(is_inputs_tuple, tuple(attr))

        else:
            return coefs
