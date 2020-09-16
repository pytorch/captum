#!/usr/bin/env python3
import warnings
from typing import Any, Callable, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from captum.log import log_usage

from ..._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_input,
    _format_output,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from ..._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from .._utils.attribution import PerturbationAttribution
from .._utils.common import _construct_default_feature_mask, _format_input_baseline


class LimeBase(PerturbationAttribution):
    r"""
    Lime is an interpretability method that trains an interpretable surrogate model
    by sampling points around a specified input example and using model evaluations
    at these points to train a simpler interpretable 'surrogate' model, such as a
    linear model.

    LimeBase provides a generic framework to train a surrogate interpretable model.
    This differs from most other attribution methods, since the method returns a
    representation of the interpretable model (e.g. coefficients of the linear model).
    For a similar interface to other perturbation-based attribution methods, please use
    the Lime child class, which defines specific transformations for the interpretable
    model.

    LimeBase allows sampling points in either the interpretable space or the original
    input space to train the surrogate model. The interpretable space is a feature
    vector used to train the surrogate interpretable model; this feature space is often
    of smaller dimensionality than the original feature space in order for the surrogate
    model to be more interpretable.

    If sampling in the interpretable space, a transformation function must be provided
    to define how a vector sampled in the interpretable space can be transformed into
    an example in the original input space. If sampling in the original input space, a
    transformation function must be provided to define how the input can be transformed
    into its interpretable vector representation.

    More details regarding LIME can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(
        self,
        forward_func: Callable,
        train_interpretable_model_func: Callable,
        similarity_func: Callable,
        sampling_func: Callable,
        sample_interpretable_space: bool,
        from_interp_rep_transform: Optional[Callable],
        to_interp_rep_transform: Optional[Callable],
    ) -> None:
        r"""

        Args:


            forward_func (callable):  The forward function of the model or any
                    modification of it
            train_interpretable_model_func (callable): Function which trains
                    an interpretable model and returns some representation of the
                    interpretable model. The return type of this will match the
                    returned type when calling attribute.
                    The expected signature of this callable is:

                    train_interpretable_model_func(
                        interp_inputs: Tensor [2D num_samples x num_interp_features],
                        exp_outputs: Tensor [1D num_samples],
                        weights: Tensor [1D num_samples]
                        **kwargs: Any
                    ) -> Any (Representation of interpretable model)

                    Note that kwargs include all kwargs passed to the attribute
                    method.
            similarity_func (callable): Function which takes a single sample
                    along with its corresponding interpretable representation
                    and returns the weight of the interpretable sample for
                    training interpretable model. Weight is generally
                    determined based on similarity to the original input.
                    The original paper refers to this as a similarity kernel.

                    The expected signature of this callable is:

                    similarity_func(
                        original_input: Tensor or tuple of Tensors,
                        sampled_input: Tensor or tuple of Tensors,
                        sampled_interpretable_input:
                            Tensor [2D 1 x num_interp_features],
                        **kwargs: Any
                    ) -> float or Tensor containing float scalar

                    sampled_input and original_input will be the same type and
                    contain tensors of the same shape (regardless of whether
                    the sampling function returns inputs in the interpretable
                    space). original_input is the same as the input provided
                    when calling attribute.

                    Note that kwargs include all kwargs passed to the attribute
                    method.
            sampling_func (callable): Function which returns a single
                    sampled input, generally a perturbation of the original
                    input, which is used to train the interpretable surrogate
                    model. Sampling function can return samples in either
                    the original input space (matching type and tensor shapes
                    of original input) or in the interpretable input space,
                    which is a vector containing the intepretable features.

                    The expected signature of this callable is:

                    sampling_func(
                        original_input: Tensor or tuple of Tensors,
                        **kwargs: Any
                    ) -> Tensor or tuple of Tensors

                    Note that kwargs include all kwargs passed to the attribute
                    method.

                    Returned sampled input should match the input type (Tensor
                    or Tuple of Tensor and corresponding shapes) if
                    sample_interpretable_space = False. If
                    sample_interpretable_space = True, the return type should
                    be a single tensor of shape 1 x num_interp_features,
                    corresponding to the representation of the
                    sample to train the interpretable model.

                    Note that kwargs include all kwargs passed to the attribute
                    method.
            sample_interpretable_space (bool, optional): Indicates whether
                    sampling_func returns a sample in the interpretable space
                    (tensor of shape 1 x num_interp_features) or a sample
                    in the original space, matching the format of the original
                    input. Once sampled, inputs can be converted to / from
                    the interpretable representation with either
                    to_interp_rep_transform or from_interp_rep_transform.
            from_interp_rep_transform (callable): Function which takes a
                    single sample's interpretable representation (tensor
                    of shape 1 x num_interp_features) and returns
                    the corresponding input in the original input space
                    (mathing shapes of original input to attribute).

                    This argument is necessary if sample_interpretable_space
                    is True, otherwise None can be provided for this argument.

                    The expected signature of this callable is:

                    from_interp_rep_transform(
                        curr_sample: Tensor [2D 1 x num_interp_features]
                        original_input: Tensor or Tuple of Tensors,
                        **kwargs: Any
                    ) -> Tensor or tuple of Tensors

                    Returned sampled input should match the type of original_input
                    and corresponding tensor shapes.

                    Note that kwargs include all kwargs passed to the attribute
                    method.

            to_interp_rep_transform (callable): Function which takes a
                    sample in the original input space and converts to
                    its interpretable representation (tensor
                    of shape 1 x num_interp_features) and returns
                    the corresponding input in the original input space.

                    This argument is necessary if sample_interpretable_space
                    is False, otherwise None can be provided for this argument.

                    The expected signature of this callable is:

                    to_interp_rep_transform(
                        curr_sample: Tensor or Tuple of Tensors,
                        original_input: Tensor or Tuple of Tensors,
                        **kwargs: Any
                    ) -> Tensor [2D 1 x num_interp_features]

                    curr_sample will match the type of original_input
                    and corresponding tensor shapes.

                    Note that kwargs include all kwargs passed to the attribute
                    method.
        """
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
            ), "Must provide transform from interpretable space to original input space"
            " when sampling from interpretable space."
        else:
            assert (
                self.to_interp_rep_transform is not None
            ), "Must provide transform from original input space to interpretable space"

    # The following overloaded method signatures correspond to the case where
    # return_convergence_delta is False, then only attributions are returned,
    # and when return_convergence_delta is True, the return type is
    # a tuple with both attributions and deltas.
    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        **kwargs
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        Nevertheless, a batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, the
        interpretable feature representation should still have shape
        1 x num_interp_features, corresponding to the interpretable
        representation for the full batch. If forward_func returns
        a single value per batch (e.g. loss), then an interpretable model is
        trained with the batch output and corresponding interpretable feature
        vector for the batch as input. If a scalar is returned per example,
        then the interpretable feature vector for the batch is repeated and
        included with each corresponding example output for interpretable
        model training.

        Args:

            inputs (tensor or tuple of tensors):  Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_samples (int, optional):  The number of samples of the original
                        model used to train the surrogate interpretabe model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            **kwargs (Any, optional): Any additional arguments necessary for
                        sampling and transformation functions (provided to
                        constructor).
                        Default: None

        Returns:
            **interpretable model representation**:
            - **interpretable model representation* (*Any*):
                    A representation of the interpretable model trained. The return
                    type matches the return type of train_interpretable_model_func.
                    For example, this could contain coefficients of a
                    linear surrogate model.

        Examples::

            >>> # SimpleClassifier takes a single input tensor of
            >>> # float features with size N x 5,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>>
            >>> # We will train an interpretable model with the same
            >>> # features by simply sampling with added Gaussian noise
            >>> # to the inputs and training a model to predict the
            >>> # score of the target class.
            >>>
            >>> # For interpretable model training, we will use sklearn
            >>> # in this example
            >>> from sklearn import linear_model
            >>>
            >>> # Define interpretable model training function
            >>> def linear_regression_interpretable_model_trainer(
            >>>     interp_inputs: Tensor,
            >>>     exp_outputs: Tensor,
            >>>     weights: Tensor, **kwargs):
            >>>         clf = linear_model.LinearRegression()
            >>>         clf.fit(
            >>>             interp_inputs.cpu().numpy(),
            >>>             exp_outputs.cpu().numpy(),
            >>>             weights.cpu().numpy())
            >>>         return clf.coef_
            >>>
            >>>
            >>> # Define similarity kernel (exponential kernel based on L2 norm)
            >>> def similarity_kernel(
            >>>     original_input: Tensor,
            >>>     sampled_input: Tensor,
            >>>     sampled_interpretable_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         # kernel_width will be provided to attribute as a kwarg
            >>>         kernel_width = kwargs["kernel_width"]
            >>>         l2_dist = torch.norm(original_input - sampled_input)
            >>>         return torch.exp(- (l2_dist**2) / (kernel_width**2))
            >>>
            >>>
            >>> # Define sampling function
            >>> # This function samples in original input space
            >>> def sampling_func(
            >>>     original_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         return original_input + torch.randn_like(original_input)
            >>>
            >>> # For this example, we are setting the interpretable input to
            >>> # match the model input, so the to_interp_rep_transform
            >>> # function simply returns the input. In most cases, the interpretable
            >>> # input will be different and may have a smaller feature set, so
            >>> # an appropriate transformation function should be provided.
            >>>
            >>> # Generating random input with size 2 x 5
            >>> input = torch.randn(2, 5)
            >>> # Defining LimeBase interpreter
            >>> lime_attr = LimeBase(net,
                                     linear_regression_interpretable_model_trainer,
                                     similarity_func=similarity_kernel,
                                     sampling_func=sampling_func,
                                     sample_interpretable_space=False,
                                     from_interp_rep_transform=None,
                                     to_interp_rep_transform=lambda x: x)
            >>> # Computes interpretable model, returning coefficients of linear
            >>> # model.
            >>> attr_coefs = lime_attr.attribute(input, target=1, kernel_width=1.1)
        """
        with torch.no_grad():
            bsz = (
                cast(Tensor, inputs).shape[0]
                if isinstance(inputs, Tensor)
                else inputs[0].shape[0]
            )
            expand_inputs = False

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
                        self.from_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                    curr_sim = self.similarity_func(
                        inputs, curr_inputs[-1], curr_sample, **kwargs
                    )
                    similarities.append(
                        curr_sim.flatten()
                        if isinstance(curr_sim, Tensor)
                        else torch.tensor([curr_sim])
                    )
                else:
                    curr_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                    curr_sim = self.similarity_func(
                        inputs, curr_sample, interpretable_inps[-1], **kwargs
                    )
                    similarities.append(
                        curr_sim.flatten()
                        if isinstance(curr_sim, Tensor)
                        else torch.tensor([curr_sim])
                    )

                if len(curr_inputs) == perturbations_per_eval:
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

                    model_out = _run_forward(
                        self.forward_func,
                        _reduce_list(curr_inputs),
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
                        ), "Number of outputs is not appropriate, must return"
                        " one output per example. If forward function returns a"
                        " scalar per batch, ensure that perturbations_per_eval is"
                        " set to 1."
                        expand_inputs = True
                    outputs.append(
                        model_out.flatten()
                        if isinstance(model_out, Tensor)
                        else torch.tensor([model_out])
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
                    model_out.flatten()
                    if isinstance(model_out, Tensor)
                    else torch.tensor([model_out])
                )

                if (
                    not expand_inputs
                    and isinstance(model_out, Tensor)
                    and model_out.numel() != len(curr_inputs)
                ):
                    assert model_out.numel() == bsz * len(
                        curr_inputs
                    ), "Number of outputs is not appropriate, must return"
                    " one output per example. If forward function returns a"
                    " scalar per batch, ensure that perturbations_per_eval is set to 1."
                    expand_inputs = True

            combined_interp_inps = torch.cat(interpretable_inps)
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            )
            combined_sim = (
                torch.cat(similarities)
                if len(similarities[0].shape) > 0
                else torch.stack(similarities)
            )
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
        import sklearn
        from sklearn import linear_model

        assert (
            sklearn.__version__ >= "0.23.0"
        ), "Must have sklearn version 0.23.0 or higher to use "
        "sample_weight in Lasso regression."
    except ImportError:
        raise AssertionError(
            "Requires sklearn for default interpretable model training with"
            " Lasso regression. Please install sklearn or use a custom interpretable"
            " model training function."
        )
    clf = linear_model.Lasso(alpha=kwargs["alpha"] if "alpha" in kwargs else 1.0)
    clf.fit(
        interp_inputs.cpu().numpy(), exp_outputs.cpu().numpy(), weights.cpu().numpy()
    )
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
    return 1.0


def default_sampling_func(original_inp, **kwargs):
    assert (
        "num_interp_features" in kwargs
    ), "Must provide num_interp_features to use default interpretable sampling function"
    if isinstance(original_inp, Tensor):
        device = original_inp.device
    else:
        device = original_inp[0].device

    probs = torch.ones(1, kwargs["num_interp_features"]) * 0.5
    return torch.bernoulli(probs).to(device=device)


class Lime(LimeBase):
    r"""
    Lime is an interpretability method that trains an interpretable surrogate model
    by sampling points around a specified input example and using model evaluations
    at these points to train a simpler interpretable 'surrogate' model, such as a
    linear model.

    Lime provides a more specific implementation than LimeBase in order to expose
    a consistent API with other perturbation-based algorithms. For more general
    use of the LIME framework, consider using the LimeBase class directly and
    defining custom sampling and transformation to / from interpretable
    representation functions.

    Lime assumes that the interpretable representation is a binary vector,
    corresponding to some elements in the input being set to their baseline value
    if the corresponding binary interpretable feature value is 0 or being set
    to the original input value if the corresponding binary interpretable
    feature value is 1. Input values can be grouped to correspond to the same
    binary interpretable feature using a feature mask provided when calling
    attribute, similar to other perturbation-based attribution methods.

    One example of this setting is when applying Lime to an image classifier.
    Pixels in an image can be grouped into super-pixels or segments, which
    correspond to interpretable features, provided as a feature_mask when
    calling attribute. Sampled binary vectors convey whether a super-pixel
    is on (retains the original input values) or off (set to the corresponding
    baseline value, e.g. black image). An interpretable linear model is trained
    with input being the binary vectors and outputs as the corresponding scores
    of the image classifier with the appropriate super-pixels masked based on the
    binary vector. Coefficients of the trained surrogate
    linear model convey the importance of each super-pixel.

    More details regarding LIME can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(
        self,
        forward_func: Callable,
        train_interpretable_model_func: Callable = lasso_interpretable_model_trainer,
        similarity_func: Callable = default_similarity_kernel,
        sampling_func: Callable = default_sampling_func,
    ) -> None:
        r"""

        Args:


            forward_func (callable):  The forward function of the model or any
                    modification of it
            train_interpretable_model_func (optional, callable): Function which
                    trains an interpretable model and returns coefficients
                    of the interpretable model.
                    This function is optional, and the default function trains
                    an interpretable model using Lasso regression, using the
                    alpha parameter provided when calling attribute.
                    Using the default function requires having sklearn version
                    0.23.0 or higher installed.

                    If a custom function is provided, the expected signature of this
                    callable is:

                    train_interpretable_model_func(
                        interp_inputs: Tensor [2D num_samples x num_interp_features],
                        exp_outputs: Tensor [1D num_samples],
                        weights: Tensor [1D num_samples]
                        **kwargs: Any
                    ) -> Tensor [1D num_interp_features]
                    The return type must be a 1D tensor containing the importance
                    or attribution of each input feature.

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask), and
                    alpha (for Lasso regression).
            similarity_func (callable): Function which takes a single sample
                    along with its corresponding interpretable representation
                    and returns the weight of the interpretable sample for
                    training the interpretable model.
                    This is often referred to as a similarity kernel.

                    The expected signature of this callable is:

                    similarity_func(
                        original_input: Tensor or tuple of Tensors,
                        sampled_input: Tensor or tuple of Tensors,
                        sampled_interpretable_input:
                            Tensor [2D 1 x num_interp_features],
                        **kwargs: Any
                    ) -> float or Tensor containing float scalar

                    sampled_input and original_input will be the same type and
                    contain tensors of the same shape, with original_input
                    being the same as the input provided when calling attribute.

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask), and
                    alpha (for Lasso regression).
            sampling_func (callable): Function which returns a single
                    sampled input, which is a binary vector of length
                    num_interp_features. The default function returns
                    a binary vector where each element is selected
                    independently and uniformly at random. Custom
                    logic for selecting sampled binary vectors can
                    be implemented by providing a function with the
                    following expected signature:

                    sampling_func(
                        original_input: Tensor or tuple of Tensors,
                        **kwargs: Any
                    ) -> Tensor [Binary 2D Tensor 1 x num_interp_features]

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask), and
                    alpha (for Lasso regression).

        """
        LimeBase.__init__(
            self,
            forward_func,
            train_interpretable_model_func,
            similarity_func,
            sampling_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        alpha: float = 1.0,
        return_input_shape: bool = True,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, the
        interpretable feature representation should still have shape
        1 x num_interp_features, corresponding to the interpretable
        representation for the full batch. If forward_func returns
        a single value per batch (e.g. loss), then an interpretable model is
        trained with the batch output and corresponding interpretable feature
        vector for the batch as input. If a scalar is returned per example,
        then the interpretable feature vector for the batch is repeated and
        included with each corresponding example output for interpretable
        model training.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (tensor or tuple of tensors):  Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                            exactly the same dimensions as inputs or the first
                            dimension is one and the remaining dimensions match
                            with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                            be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                            to each tensor in the inputs' tuple can be:

                            - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                            - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.
                        In the cases when `baselines` is not provided, we internally
                        use zero corresponding to each input tensor.
                        Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            feature_mask (tensor or tuple of tensors, optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same
                        interpretable feature. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Values across
                        all tensors should be integers in the range 0 to
                        num_interp_features - 1, and indices corresponding to the
                        same feature should have the same value.
                        Note that features are grouped across tensors
                        (unlike feature ablation and occlusion), so
                        if the same index is used in different tensors, those
                        features are still grouped and added simultaneously.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature.
                        Default: None
            n_samples (int, optional):  The number of samples of the original
                        model used to train the surrogate interpretabe model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            alpha (float, optional):  Alpha used for training interpretable surrogate
                        model in Lasso Regression. This parameter is used only
                        if using default interpretable model trainer (Lasso).
                        Default: 1.0
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each interp-
                        retable feature from the trained surrogate model, or
                        whether the returned attributions match the input shape.
                        When return_input_shape is True, the return type of attribute
                        matches the input shape, with each element containing the
                        coefficient of the corresponding interpretale feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions. If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        The attributions with respect to each input feature.
                        If return_input_shape = True, attributions will be
                        the same size as the provided inputs, with each value
                        providing the coefficient of the corresponding
                        interpretale feature.
                        If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.
        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4
            >>> input = torch.randn(1, 4, 4)

            >>> # Defining Lime interpreter
            >>> lime = Lime(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = lime.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we can group each 2x2 square of the inputs
            >>> # as one 'interpretable' feature and perturb them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are set to their
            >>> # baseline value, when the corresponding binary interpretable
            >>> # feature is set to 0.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # Computes interpretable model and returning attributions
            >>> # matching input shape.
            >>> attr = lime.attribute(input, target=1, feature_mask=feature_mask)
        """
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)

        if feature_mask is None:
            feature_mask, num_interp_features = _construct_default_feature_mask(
                formatted_inputs
            )
        else:
            feature_mask = _format_input(feature_mask)
            num_interp_features = int(
                max(torch.max(single_inp).item() for single_inp in feature_mask) + 1
            )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. "
            )

        coefs = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            alpha=alpha,
        )
        if return_input_shape:
            attr = [torch.zeros_like(inp) for inp in formatted_inputs]
            for tensor_ind in range(len(formatted_inputs)):
                for single_feature in range(num_interp_features):
                    attr[tensor_ind] += (
                        coefs[single_feature].item()
                        * (feature_mask[tensor_ind] == single_feature).float()
                    )
            return _format_output(is_inputs_tuple, tuple(attr))

        else:
            return coefs
