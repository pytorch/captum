#!/usr/bin/env python3

from typing import Callable, Optional, Tuple, Union, Any, List

import torch
import torch.nn as nn
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
    total = torch.Tensor(total)

    return total


def _jacobian_loss_wrt_inputs(
    loss_fn: Union[Module, Callable], out: Tensor, targets: Tensor, vectorize: bool
) -> Tensor:
    r"""
    Helper function to handle dealing with pytorch version differences for vectorized
    jacobian calculation of loss wrt inputs.
    """

    if torch.__version__ >= "1.8":
        return torch.autograd.functional.jacobian(
            lambda out: loss_fn(out, targets), out, vectorize=vectorize
        )
    else:
        return torch.autograd.functional.jacobian(
            lambda out: loss_fn(out, targets), out
        )


def _load_flexible_state_dict(
    model: Module, path: str, device_ids: str = "cpu", keyname: Optional[str] = None
) -> int:
    r"""
    Helper to load pytorch models. This function attempts to find compatibility for
    loading models that were trained on different devices / with DataParallel but are
    being loaded in a different environment.

    Assumes that the model has been saved as a state_dict in some capacity. This can
    either be a single state dict, or a nesting dictionary which contains the model
    state_dict and other information.

    Args:
        model: The model for which to load a checkpoint
        path: The filepath to the checkpoint
        keyname: The key under which the model state_dict is stored, if any.

    The module state_dict is modified in-place, and the learning rate is returned.
    """

    device = device_ids

    checkpoint = torch.load(path, map_location=device)

    learning_rate = checkpoint.get("learning_rate", 1)
    # can get learning rate from optimizer state_dict?

    if keyname is not None:
        checkpoint = checkpoint[keyname]

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


def _format_inputs(inputs: Any, unpack_inputs: bool):
    if inputs is None:
        return None
    return (
        inputs
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and unpack_inputs
        else (inputs,)
    )


def _get_k_most_influential_helper(
    influence_src_dataloader: DataLoader,
    influence_batch_fn: Callable,
    inputs: Tuple[Any, ...],
    targets: Optional[Tensor],
    k: int = 5,
    proponents: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""
    Helper function that computes the quantities returned by
    `TracInCPBase._get_k_most_influential`, using a specific implementation that is
    constant memory.

    Args:
        influence_src_dataloader (DataLoader): The DataLoader, representing training
                data, for which we want to compute proponents / opponents.
        influence_batch_fn (Callable): A callable that will be called via
                `influence_batch_fn(inputs, targets, batch)`, where `batch` is a batch
                in the `influence_src_dataloader` argument.
        inputs (Tuple of Any): A batch of examples. Does not represent labels,
                which are passed as `targets`.
        targets (Tensor, optional): If computing TracIn scores on a loss function,
                these are the labels corresponding to the batch `inputs`.
                Default: None
        k (int, optional): The number of proponents or opponents to return per test
                instance.
                Default: 5
        proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                or opponents (`proponents=False`)
                Default: True

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

    for batch in influence_src_dataloader:

        # calculate tracin_scores for the batch
        batch_tracin_scores = influence_batch_fn(inputs, targets, batch)
        batch_tracin_scores *= multiplier

        # get the top-k indices and tracin_scores for the batch
        batch_size = batch_tracin_scores.shape[1]
        batch_topk_tracin_scores, batch_topk_indices = torch.topk(
            batch_tracin_scores, min(batch_size, k), dim=1
        )
        batch_topk_indices = batch_topk_indices + num_instances_processed
        num_instances_processed += batch_size

        # combine the top-k for the batch with those for previously seen batches
        topk_indices = torch.cat([topk_indices, batch_topk_indices], dim=1)
        topk_tracin_scores = torch.cat(
            [topk_tracin_scores, batch_topk_tracin_scores], dim=1
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
    def __init__(self, _l: List[Any]):
        self._l = _l

    def __getitem__(self, i: int) -> Any:
        return self._l[i]

    def __len__(self) -> int:
        return len(self._l)
