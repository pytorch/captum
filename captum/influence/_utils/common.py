#!/usr/bin/env python3
import warnings
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn
from captum._utils.common import _parse_version
from captum._utils.progress import progress

if TYPE_CHECKING:
    from captum.influence._core.tracincp import TracInCPBase

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


def _tensor_batch_dot(t1: Tensor, t2: Tensor) -> Tensor:
    r"""
    Computes pairwise dot product between two tensors

    Args:
        Tensors t1 and t2 are feature vectors with dimension (batch_size_1, *) and
        (batch_size_2,  *). The * dimensions must match in total number of elements.

    Returns:
        Tensor with shape (batch_size_1, batch_size_2) containing the pairwise dot
        products. For example, Tensor[i][j] would be the dot product between
        t1[i] and t2[j].
    """

    msg = (
        "Please ensure each batch member has the same feature dimension. "
        f"First input has {torch.numel(t1) / t1.shape[0]} features, and "
        f"second input has {torch.numel(t2) / t2.shape[0]} features."
    )
    assert torch.numel(t1) / t1.shape[0] == torch.numel(t2) / t2.shape[0], msg

    return torch.mm(
        t1.view(t1.shape[0], -1),
        t2.view(t2.shape[0], -1).T,
    )


def _gradient_dot_product(
    input_grads: Tuple[Tensor], src_grads: Tuple[Tensor]
) -> Tensor:
    r"""
    Computes the dot product between the gradient vector for a model on an input batch
    and src batch, for each pairwise batch member. Gradients are passed in as a tuple
    corresponding to the trainable parameters returned by model.parameters(). Output
    corresponds to a tensor of size (inputs_batch_size, src_batch_size) with all
    pairwise dot products.
    """

    assert len(input_grads) == len(src_grads), "Mismatching gradient parameters."

    iterator = zip(input_grads, src_grads)
    total = _tensor_batch_dot(*next(iterator))
    for input_grad, src_grad in iterator:
        total += _tensor_batch_dot(input_grad, src_grad)

    return total


def _jacobian_loss_wrt_inputs(
    loss_fn: Union[Module, Callable],
    out: Tensor,
    targets: Tensor,
    vectorize: bool,
    reduction_type: str,
) -> Tensor:
    r"""
    Often, we have a loss function that computes a per-sample loss given a 1D tensor
    input, and we want to calculate the jacobian of the loss w.r.t. that input.  For
    example, the input could be a length K tensor specifying the probability a given
    sample belongs to each of K possible classes, and the loss function could be
    cross-entropy loss. This function performs that calculation, but does so for a
    *batch* of inputs. We create this helper function for two reasons: 1) to handle
    differences between Pytorch versiosn for vectorized jacobian calculations, and
    2) this function does not accept the aforementioned per-sample loss function.
    Instead, it accepts a "reduction" loss function that *reduces* the per-sample loss
    for a batch into a single loss. Using a "reduction" loss improves speed.
    We will allow this reduction to either be the mean or sum of the per-sample losses,
    and this function provides an uniform way to handle different possible reductions,
    and also check if the reduction used is valid. Regardless of the reduction used,
    this function returns the jacobian for the per-sample loss (for each sample in the
    batch).

    Args:
        loss_fn (torch.nn.Module, Callable, or None): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='sum'`
                or `reduction='mean'`.
        out (Tensor): This is a tensor that represents the batch of inputs to
                `loss_fn`. In practice, this will be the output of a model; this is
                why this argument is named `out`. `out` is a 2D tensor of shape
                (batch size, model output dimensionality). We will call `loss_fn` via
                `loss_fn(out, targets)`.
        targets (Tensor): The labels for the batch of inputs.
        vectorize (bool): Flag to use experimental vectorize functionality for
                `torch.autograd.functional.jacobian`.
        reduction_type (str): The type of reduction used by `loss_fn`. If `loss_fn`
                has the "reduction" attribute, we will check that they match. Can
                only be "mean" or "sum".

    Returns:
        jacobians (Tensor): Returns the jacobian of the per-sample loss (implicitly
                defined by `loss_fn` and `reduction_type`) w.r.t each sample
                in the batch represented by `out`. This is a 2D tensor, where the
                first dimension is the batch dimension.
    """
    # TODO: allow loss_fn to be Callable
    if isinstance(loss_fn, Module) and hasattr(loss_fn, "reduction"):
        msg0 = "Please ensure that loss_fn.reduction is set to `sum` or `mean`"

        assert loss_fn.reduction != "none", msg0
        msg1 = (
            f"loss_fn.reduction ({loss_fn.reduction}) does not match"
            f"reduction type ({reduction_type}). Please ensure they are"
            " matching."
        )
        assert loss_fn.reduction == reduction_type, msg1

    if reduction_type != "sum" and reduction_type != "mean":
        raise ValueError(
            f"{reduction_type} is not a valid value for reduction_type. "
            "Must be either 'sum' or 'mean'."
        )

    if _parse_version(torch.__version__) >= (1, 8, 0):
        input_jacobians = torch.autograd.functional.jacobian(
            lambda out: loss_fn(out, targets), out, vectorize=vectorize
        )
    else:
        input_jacobians = torch.autograd.functional.jacobian(
            lambda out: loss_fn(out, targets), out
        )

    if reduction_type == "mean":
        input_jacobians = input_jacobians * len(input_jacobians)

    return input_jacobians


def _load_flexible_state_dict(model: Module, path: str) -> float:
    r"""
    Helper to load pytorch models. This function attempts to find compatibility for
    loading models that were trained on different devices / with DataParallel but are
    being loaded in a different environment.

    Assumes that the model has been saved as a state_dict in some capacity. This can
    either be a single state dict, or a nesting dictionary which contains the model
    state_dict and other information.

    Args:

        model (torch.nn.Module): The model for which to load a checkpoint
        path (str): The filepath to the checkpoint

    The module state_dict is modified in-place, and the learning rate is returned.
    """

    checkpoint = torch.load(path)

    learning_rate = checkpoint.get("learning_rate", 1.0)
    # can get learning rate from optimizer state_dict?

    if "module." in next(iter(checkpoint)):
        if isinstance(model, nn.DataParallel):
            model.load_state_dict(checkpoint)
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint)
            model = model.module
    else:
        if isinstance(model, nn.DataParallel):
            model = model.module
            model.load_state_dict(checkpoint)
            model = nn.DataParallel(model)
        else:
            model.load_state_dict(checkpoint)

    return learning_rate


def _get_k_most_influential_helper(
    influence_src_dataloader: DataLoader,
    influence_batch_fn: Callable,
    inputs: Tuple[Any, ...],
    k: int = 5,
    proponents: bool = True,
    show_progress: bool = False,
    desc: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Helper function that computes the quantities returned by
    `TracInCPBase._get_k_most_influential`, using a specific implementation that is
    constant memory.

    Args:
        influence_src_dataloader (DataLoader): The DataLoader, representing training
                data, for which we want to compute proponents / opponents.
        influence_batch_fn (Callable): A callable that will be called via
                `influence_batch_fn(inputs, batch)`, where `batch` is a batch
                in the `influence_src_dataloader` argument.
        inputs (tuple[Any, ...]): This argument represents the test batch, and is a
                single tuple of any, where the last element is assumed to be the labels
                for the batch. That is, `model(*batch[0:-1])` produces the output for
                `model`, and `batch[-1]` are the labels, if any.
        k (int, optional): The number of proponents or opponents to return per test
                instance.
                Default: 5
        proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                or opponents (`proponents=False`)
                Default: True
        show_progress (bool, optional): To compute the proponents (or opponents)
                for the batch of examples, we perform computation for each batch in
                training dataset `influence_src_dataloader`, If `show_progress` is
                true, the progress of this computation will be displayed. In
                particular, the number of batches for which the computation has
                been performed will be displayed. It will try to use tqdm if
                available for advanced features (e.g. time estimation). Otherwise,
                it will fallback to a simple output of progress.
                Default: False
        desc (str, optional): If `show_progress` is true, this is the description to
                show when displaying progress. If `desc` is none, no description is
                shown.
                Default: None

    Returns:
        (indices, influence_scores): `indices` is a torch.long Tensor that contains the
                indices of the proponents (or opponents) for each test example. Its
                dimension is `(inputs_batch_size, k)`, where `inputs_batch_size` is the
                number of examples in `inputs`. For example, if `proponents==True`,
                `indices[i][j]` is the index of the example in training dataset
                `influence_src_dataloader` with the k-th highest influence score for
                the j-th example in `inputs`. `indices` is a `torch.long` tensor so that
                it can directly be used to index other tensors. Each row of
                `influence_scores` contains the influence scores for a different test
                example, in sorted order. In particular, `influence_scores[i][j]` is
                the influence score of example `indices[i][j]` in training dataset
                `influence_src_dataloader` on example `i` in the test batch represented
                by `inputs` and `targets`.
    """
    # For each test instance, maintain the best indices and corresponding distances
    # initially, these will be empty
    topk_indices = torch.Tensor().long()
    topk_tracin_scores = torch.Tensor()

    multiplier = 1.0 if proponents else -1.0

    # needed to map from relative index in a batch fo index within entire `dataloader`
    num_instances_processed = 0

    # if show_progress, create progress bar
    total: Optional[int] = None
    if show_progress:
        try:
            total = len(influence_src_dataloader)
        except AttributeError:
            pass
        influence_src_dataloader = progress(
            influence_src_dataloader,
            desc=desc,
            total=total,
        )

    for batch in influence_src_dataloader:

        # calculate tracin_scores for the batch
        batch_tracin_scores = influence_batch_fn(inputs, batch)
        batch_tracin_scores *= multiplier

        # get the top-k indices and tracin_scores for the batch
        batch_size = batch_tracin_scores.shape[1]
        batch_topk_tracin_scores, batch_topk_indices = torch.topk(
            batch_tracin_scores, min(batch_size, k), dim=1
        )
        batch_topk_indices = batch_topk_indices + num_instances_processed
        num_instances_processed += batch_size

        # combine the top-k for the batch with those for previously seen batches
        topk_indices = torch.cat(
            [topk_indices.to(batch_topk_indices.device), batch_topk_indices], dim=1
        )
        topk_tracin_scores = torch.cat(
            [
                topk_tracin_scores.to(batch_topk_tracin_scores.device),
                batch_topk_tracin_scores,
            ],
            dim=1,
        )

        # retain only the top-k in terms of tracin_scores
        topk_tracin_scores, topk_argsort = torch.topk(
            topk_tracin_scores, min(k, topk_indices.shape[1]), dim=1
        )
        topk_indices = torch.gather(topk_indices, dim=1, index=topk_argsort)

    # if seeking opponents, we were actually keeping track of negative tracin_scores
    topk_tracin_scores *= multiplier

    return topk_indices, topk_tracin_scores


class _DatasetFromList(Dataset):
    def __init__(self, _l: List[Any]) -> None:
        self._l = _l

    def __getitem__(self, i: int) -> Any:
        return self._l[i]

    def __len__(self) -> int:
        return len(self._l)


def _format_inputs_dataset(inputs_dataset: Union[Tuple[Any, ...], DataLoader]):
    # if `inputs_dataset` is not a `DataLoader`, turn it into one.
    # `_DatasetFromList` turns a list into a `Dataset` where `__getitem__`
    # returns an element in the list, and using it to construct a `DataLoader`
    # with `batch_size=None` gives a `DataLoader` that yields a single batch.
    if not isinstance(inputs_dataset, DataLoader):
        inputs_dataset = DataLoader(
            _DatasetFromList([inputs_dataset]), shuffle=False, batch_size=None
        )
    return inputs_dataset


def _self_influence_by_batches_helper(
    self_influence_batch_fn: Callable,
    instance_name: str,
    inputs_dataset: Union[Tuple[Any, ...], DataLoader],
    show_progress: bool = False,
) -> Tensor:
    """
    Computes self influence scores for the examples in `inputs_dataset`, which is
    either a single batch or a Pytorch `DataLoader` that yields batches. The self
    influence scores for a single batch are computed using the
    `self_influence_batch_fn` input. Note that if `inputs_dataset` is a single batch,
    this will call `model` on that single batch, where `model` is the model used to
    compute self influence scores by `self_influence_batch_fn`, and if `inputs_dataset`
    yields batches, this will call `model` on each batch that is yielded. Therefore,
    please ensure that for both cases, the batch(es) that `model` is called
    with are not too large, so that there will not be an out-of-memory error. This
    implementation performs an outer iteration over all batches that
    `inputs_dataset` represents, and an inner iteration over checkpoints. The pros
    of this implementation are that showing the progress of the computation is
    straightforward.

    Args:
        self_influence_batch_fn (Callable): This is the function that computes self
                influence scores for a single batch.
        instance_name (str): This is the name of the implementation class that
                `self_influence_batch_fn` is a method of. This is used for displaying
                warning messages.
        batches (tuple or DataLoader): Either a single tuple of any, or a
                `DataLoader`, where each batch yielded is a tuple of any. In
                either case, the tuple represents a single batch, where the last
                element is assumed to be the labels for the batch. That is,
                `model(*batch[0:-1])` produces the output for `model`,
                and `batch[-1]` are the labels, if any. This is the same
                assumption made for each batch yielded by training dataset
                `train_dataset`. Please see documentation for the
                `train_dataset` argument to `TracInCP.__init__` for
                more details on the assumed structure of a batch.
        show_progress (bool, optional): Computation of self influence scores can
                take a long time if `inputs_dataset` represents many examples. If
                `show_progress`is true, the progress of this computation will be
                displayed. In particular, the number of batches for which self
                influence scores have been computed will be displayed. It will try
                to use tqdm if available for advanced features (e.g. time
                estimation). Otherwise, it will fallback to a simple output of
                progress.
                Default: False

    Returns:
        self_influence_scores (Tensor): This is a 1D tensor containing the self
                influence scores of all examples in `inputs_dataset`, regardless of
                whether it represents a single batch or a `DataLoader` that yields
                batches.
    """
    # If `inputs_dataset` is not a `DataLoader`, turn it into one.
    inputs_dataset = _format_inputs_dataset(inputs_dataset)

    # If `show_progress` is true, create a progress bar that keeps track of how
    # many batches have been processed
    if show_progress:
        # First, try to determine length of progress bar if possible, with a
        # default of `None`
        inputs_dataset_len = None
        try:
            inputs_dataset_len = len(inputs_dataset)
        except TypeError:
            warnings.warn(
                "Unable to determine the number of batches in `inputs_dataset`. "
                "Therefore, if showing the progress of the computation of self "
                "influence scores, only the number of batches processed can be "
                "displayed, and not the percentage completion of the computation, "
                "nor any time estimates."
            )
        # then create the progress bar
        inputs_dataset = progress(
            inputs_dataset,
            desc=f"Using {instance_name} to compute self influence. Processing batch",
            total=inputs_dataset_len,
        )

    # To compute self influence scores for each batch, we use
    # `_self_influence_by_checkpoints`, which can accept a tuple representing a
    # single batch as the `inputs_dataset` argument (as well as a DataLoader).
    # Because we are already displaying progress in terms of number of batches
    # processed in this method, we will not show progress for the call to
    # `_self_influence_by_checkpoints`.
    return torch.cat(
        [
            self_influence_batch_fn(batch, show_progress=False)
            for batch in inputs_dataset
        ]
    )


def _check_loss_fn(
    influence_instance: "TracInCPBase",
    loss_fn: Optional[Union[Module, Callable]],
    loss_fn_name: str,
    sample_wise_grads_per_batch: Optional[bool] = None,
) -> str:
    """
    This checks whether `loss_fn` satisfies the requirements assumed of all
    implementations of `TracInCPBase`. It works regardless of whether the
    implementation has the `sample_wise_grads_per_batch` attribute.
    It returns the reduction type of the loss_fn. If `sample_wise_grads_per_batch`
    if not provided, we assume the implementation does not have that attribute.
    """
    # if `loss_fn` is `None`, there is nothing to check. then, the reduction type is
    # only used by `_compute_jacobian_wrt_params_with_sample_wise_trick`, where
    # reduction type should be "sum" if `loss_fn` is `None`.
    if loss_fn is None:
        return "sum"

    # perhaps since `Module` is an implementation of `Callable`, this has redundancy
    assert isinstance(loss_fn, Module) or callable(loss_fn)

    reduction_type = "none"

    # If we are able to access the reduction used by `loss_fn`, we check whether
    # the reduction is compatible with `sample_wise_grads_per_batch`, if it has the
    # attribute.
    if hasattr(loss_fn, "reduction"):
        reduction = loss_fn.reduction  # type: ignore
        if sample_wise_grads_per_batch is None:
            assert reduction in [
                "sum",
                "mean",
            ], 'reduction for `loss_fn` must be "sum" or "mean"'
            reduction_type = str(reduction)
        elif sample_wise_grads_per_batch:
            assert reduction in ["sum", "mean"], (
                'reduction for `loss_fn` must be "sum" or "mean" when '
                "`sample_wise_grads_per_batch` is True"
            )
            reduction_type = str(reduction)
        else:
            assert reduction == "none", (
                'reduction for `loss_fn` must be "none" when '
                "`sample_wise_grads_per_batch` is False"
            )
    else:
        # if we are unable to access the reduction used by `loss_fn`, we warn
        # the user about the assumptions we are making regarding the reduction
        # used by `loss_fn`
        if sample_wise_grads_per_batch is None:
            warnings.warn(
                f'Since `{loss_fn_name}` has no "reduction" attribute, the '
                f'implementation  assumes that `{loss_fn_name}` is a "reduction" loss '
                "function that reduces the per-example losses by taking their *sum*. "
                f"If `{loss_fn_name}` instead reduces the per-example losses by "
                f"taking their mean, please set the reduction attribute of "
                f'`{loss_fn_name}` to "mean", i.e. '
                f'`{loss_fn_name}.reduction = "mean"`.'
            )
            reduction_type = "sum"
        elif sample_wise_grads_per_batch:
            warnings.warn(
                f"Since `{loss_fn_name}`` has no 'reduction' attribute, and "
                "`sample_wise_grads_per_batch` is True, the implementation assumes "
                f"that `{loss_fn_name}` is a 'reduction' loss function that reduces "
                f"the per-example losses by taking their *sum*. If `{loss_fn_name}` "
                "instead reduces the per-example losses by taking their mean, "
                f'please set the reduction attribute of `{loss_fn_name}` to "mean", '
                f'i.e. `{loss_fn_name}.reduction = "mean"`. Note that if '
                "`sample_wise_grads_per_batch` is True, the implementation "
                "assumes the reduction is either a sum or mean reduction."
            )
            reduction_type = "sum"
        else:
            warnings.warn(
                f'Since `{loss_fn_name}` has no "reduction" attribute, and '
                "`sample_wise_grads_per_batch` is False, the implementation "
                f'assumes that `{loss_fn_name}` is a "per-example" loss function (see '
                f"documentation for `{loss_fn_name}` for details).  Please ensure "
                "that this is the case."
            )

    return reduction_type
