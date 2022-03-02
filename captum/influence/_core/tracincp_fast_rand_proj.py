#!/usr/bin/env python3

import warnings
from typing import Any, Callable, Iterator, List, Optional, Union, Tuple

import torch
from captum._utils.common import _get_module_from_name
from captum.influence._core.tracincp import TracInCPBase, KMostInfluentialResults
from captum.influence._utils.common import (
    _jacobian_loss_wrt_inputs,
    _load_flexible_state_dict,
    _tensor_batch_dot,
    _get_k_most_influential_helper,
    _DatasetFromList,
)
from captum.influence._utils.nearest_neighbors import (
    NearestNeighbors,
    AnnoyNearestNeighbors,
)
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

layer_inputs = []


def _capture_inputs(layer: Module, input: Tensor, output: Tensor) -> None:
    r"""Save activations into layer.activations in forward pass"""

    layer_inputs.append(input[0].detach())


r"""
Implements abstract DataInfluence class and also provides implementation details for
influence computation based on the logic provided in TracIn paper
(https://arxiv.org/pdf/2002.08484.pdf).

The TracIn paper proposes an idealized notion of influence which can be represented by
the total amount a training example reduces loss for a test example via a training
process such as stochastic gradient descent. As this idealized notion of influence is
impractical to compute, the TracIn paper proposes instead to compute an influence
score, which uses a first-order approximation for the change in loss for a test example
by a training example, which is accumulated across saved model checkpoints. This
influence score is accumulated via a summed dot-product of gradient vectors for the
scores/loss of a test and training example.
"""

"""
TODO: Support for checkpoint type. Currently only supports model parameters as saved
checkpoints. Can use enum or string.

Potential implementation from design doc:
checkpoint_type (Enum = [Parameters | Loss_Grad]): For performance,
                saved / loaded checkpoints can be either model parameters, or
                gradient of the loss function on an input w.r.t parameters.
"""


class TracInCPFast(TracInCPBase):
    r"""
    In Appendix F, Page 14 of the TracIn paper, they show that the calculation
    of the influence score of between a test example x' and a training example x,
    can be computed much more quickly than naive back-propagation in the special
    case when considering only gradients in the last fully-connected layer. This class
    computes influence scores for that special case. Note that the computed
    influence scores are exactly the same as when naive back-propagation is used -
    there is no loss in accuracy.
    """

    def __init__(
        self,
        model: Module,
        final_fc_layer: Union[Module, str],
        influence_src_dataset: Union[Dataset, DataLoader],
        checkpoints: Union[str, List[str], Iterator],
        checkpoints_load_func: Callable = _load_flexible_state_dict,
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        vectorize: bool = False,
    ) -> None:
        r"""
        Args:
            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            final_fc_layer (torch.nn.Module or str): The last fully connected layer in
                    the network for which gradients will be approximated via fast random
                    projection method. Can be either the layer module itself, or the
                    fully qualified name of the layer if it is a defined attribute of
                    the passed `model`.
            influence_src_dataset (torch.utils.data.Dataset or torch.utils.DataLoader):
                    In the `influence` method, we either compute the influence score of
                    training examples on examples in a test batch, or self influence
                    scores for those training examples, depending on which mode is used.
                    This argument represents the training dataset containing those
                    training examples. In order to compute those influence scores, we
                    will create a Pytorch DataLoader yielding batches of training
                    examples that is then used for processing. If this argument is
                    already a Pytorch Dataloader, that DataLoader can be directly
                    used for processing. If it is instead a Pytorch Dataset, we will
                    create a DataLoader using it, with batch size specified by
                    `batch_size`. For efficiency purposes, the batch size of the
                    DataLoader used for processing should be as large as possible, but
                    not too large, so that certain intermediate quantities created
                    from a batch still fit in memory. Therefore, if
                    `influence_src_dataset` is a Dataset, `batch_size` should be large.
                    If `influence_src_dataset` was already a DataLoader to begin with,
                    it should have been constructed to have a large batch size.
            checkpoints (str or List of str or Iterator): Either the directory of the
                    path to store and retrieve model checkpoints, a list of
                    filepaths with checkpoints from which to load, or an iterator which
                    returns objects from which to load checkpoints.
            checkpoints_load_func (Callable, optional): The function to load a saved
                    checkpoint into a model to update its parameters, and get the
                    learning rate if it is saved. By default uses a utility to load a
                    model saved as a state dict.
                    Default: _load_flexible_state_dict
            loss_fn (Callable, optional): The loss function applied to model. `loss_fn`
                    must be a "reduction" loss function that reduces the per-example
                    losses in a batch, and returns a single scalar Tensor. Furthermore,
                    the reduction must be the *sum* of the per-example losses. For
                    instance, `nn.BCELoss(reduction="sum")` is acceptable, but
                    `nn.BCELoss(reduction="mean")` is *not* acceptable.
                    Default: None
            batch_size (int or None, optional): Batch size of the DataLoader created to
                    iterate through `influence_src_dataset`, if it is a Dataset.
                    `batch_size` should be chosen as large as possible so that certain
                    intermediate quantities created from a batch still fit in memory.
                    Specific implementations of `TracInCPBase` will detail the size of
                    the intermediate quantities. `batch_size` must be an int if
                    `influence_src_dataset` is a Dataset. If `influence_src_dataset`
                    is a DataLoader, then `batch_size` is ignored as an argument.
                    Default: 1
            vectorize (bool, optional): Flag to use experimental vectorize functionality
                    for `torch.autograd.functional.jacobian`.
                    Default: False
        """
        TracInCPBase.__init__(
            self,
            model,
            influence_src_dataset,
            checkpoints,
            checkpoints_load_func,
            loss_fn,
            batch_size,
        )

        self.vectorize = vectorize

        # TODO: restore prior state
        self.final_fc_layer = final_fc_layer
        if isinstance(self.final_fc_layer, str):
            self.final_fc_layer = _get_module_from_name(model, self.final_fc_layer)
        assert isinstance(self.final_fc_layer, Module)
        for param in self.final_fc_layer.parameters():
            param.requires_grad = True

        assert loss_fn is not None, "loss function must not be none"
        # If we are able to access the reduction used by `loss_fn`, we check whether
        # the reduction is "sum", as required.
        # TODO: allow loss_fn to be Callable
        if isinstance(loss_fn, Module) and hasattr(loss_fn, "reduction"):
            msg = "`loss_fn.reduction` must be `sum`."
            assert loss_fn.reduction == "sum", msg

    def _influence_batch_tracincp_fast(
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
        batch: Tuple[Any, ...],
    ):
        """
        computes influence scores for a single training batch
        """

        def get_checkpoint_contribution(checkpoint):

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            input_jacobians, input_layer_inputs = _basic_computation_tracincp_fast(
                self,
                inputs,
                targets,
            )

            src_jacobian, src_layer_input = _basic_computation_tracincp_fast(
                self, batch[0:-1], batch[-1]
            )
            return (
                _tensor_batch_dot(input_jacobians, src_jacobian)
                * _tensor_batch_dot(input_layer_inputs, src_layer_input)
                * learning_rate
            )

        batch_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

        for checkpoint in self.checkpoints[1:]:
            batch_tracin_scores += get_checkpoint_contribution(checkpoint)

        return batch_tracin_scores

    def _influence(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
    ) -> Tensor:
        r"""
        Computes the influence of examples in training dataset `influence_src_dataset`
        on the examples in the test batch represented by `inputs` and `targets`.
        This implementation does not require knowing the number of training examples
        in advance. Instead, the number of training examples is inferred from the
        output of `_basic_computation_tracincp_fast`.

        Args:
            inputs (Tuple of Any): A batch of examples. Does not represent labels,
                    which are passed as `targets`. The assumption is that
                    `self.model(*inputs)` produces the predictions for the batch.
            targets (tensor): The labels corresponding to the batch `inputs`. This
                    method is designed to be applied for a loss function, so labels
                    are required.

        Returns:
            influence_scores (tensor): Influence scores from the TracInCPFast method.
            Its shape is `(input_size, influence_src_dataset_size)`, where `input_size`
            is the number of examples in the test batch, and
            `influence_src_dataset_size` is the number of examples in
            training dataset `influence_src_dataset`. For example:
            `influence_scores[i][j]` is the influence score for the j-th training
            example to the i-th input example.
        """
        assert targets is not None
        return torch.cat(
            [
                self._influence_batch_tracincp_fast(inputs, targets, batch)
                for batch in self.influence_src_dataloader
            ],
            dim=1,
        )

    def _get_k_most_influential(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
        k: int = 5,
        proponents: bool = True,
    ) -> KMostInfluentialResults:
        r"""
        Args:
            inputs (Tuple of Any): A tuple that represents a batch of examples. It does
                    not represent labels, which are passed as `targets`.
            targets (tensor): The labels corresponding to the batch `inputs`. This
                    method is designed to be applied for a loss function, so labels
                    are required.
            k (int, optional): The number of proponents or opponents to return per test
                    example.
                    Default: 5
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`)
                    Default: True

        Returns:
            (indices, influence_scores) (namedtuple): `indices` is a torch.long Tensor
                    that contains the indices of the proponents (or opponents) for each
                    test example. Its dimension is `(inputs_batch_size, k)`, where
                    `inputs_batch_size` is the number of examples in `inputs`. For
                    example, if `proponents==True`, `indices[i][j]` is the index of the
                    example in training dataset `influence_src_dataset` with the
                    k-th highest influence score for the j-th example in `inputs`.
                    `indices` is a `torch.long` tensor so that it can directly be used
                    to index other tensors. Each row of `influence_scores` contains the
                    influence scores for a different test example, in sorted order. In
                    particular, `influence_scores[i][j]` is the influence score of
                    example `indices[i][j]` in training dataset `influence_src_dataset`
                    on example `i` in the test batch represented by `inputs` and
                    `targets`.
        """
        return KMostInfluentialResults(
            *_get_k_most_influential_helper(
                self.influence_src_dataloader,
                self._influence_batch_tracincp_fast,
                inputs,
                targets,
                k,
                proponents,
            )
        )

    def _self_influence_batch_tracincp_fast(self, batch: Tuple[Any, ...]):
        """
        Computes self influence scores for a single batch
        """

        def get_checkpoint_contribution(checkpoint):

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            batch_jacobian, batch_layer_input = _basic_computation_tracincp_fast(
                self, batch[0:-1], batch[-1]
            )

            return (
                torch.sum(batch_jacobian ** 2, dim=1)
                * torch.sum(batch_layer_input ** 2, dim=1)
                * learning_rate
            )

        batch_self_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

        for checkpoint in self.checkpoints[1:]:
            batch_self_tracin_scores += get_checkpoint_contribution(checkpoint)

        return batch_self_tracin_scores

    def _self_influence(self):
        """
        Returns:
            self influence scores (tensor): 1D tensor containing self influence
                    scores for all examples in training dataset
                    `influence_src_dataset`.
        """
        return torch.cat(
            [
                self._self_influence_batch_tracincp_fast(batch)
                for batch in self.influence_src_dataloader
            ],
            dim=0,
        )


def _basic_computation_tracincp_fast(
    influence_instance: TracInCPFast,
    inputs: Tuple[Any, ...],
    targets: Tensor,
):
    """
    For instances of TracInCPFast and children classes, computation of influence scores
    or self influence scores repeatedly calls this function for different checkpoints
    and batches.

    Args:
        influence_instance (TracInCPFast): A instance of TracInCPFast or its children.
        inputs (Tuple of Any): A batch of examples, which could be a training batch
                or test batch, depending which method is the caller. Does not
                represent labels, which are passed as `targets`. The assumption is
                that `self.model(*inputs)` produces the predictions for the batch.
        targets (tensor): If computing influence scores on a loss function,
                these are the labels corresponding to the batch `inputs`.
    """
    global layer_inputs
    layer_inputs = []
    assert isinstance(influence_instance.final_fc_layer, Module)
    handle = influence_instance.final_fc_layer.register_forward_hook(_capture_inputs)
    out = influence_instance.model(*inputs)

    assert influence_instance.loss_fn is not None
    input_jacobians = _jacobian_loss_wrt_inputs(
        influence_instance.loss_fn, out, targets, influence_instance.vectorize
    )
    handle.remove()
    _layer_inputs = layer_inputs[0]

    return input_jacobians, _layer_inputs


class TracInCPFastRandProj(TracInCPFast):
    def __init__(
        self,
        model: Module,
        final_fc_layer: Union[Module, str],
        influence_src_dataset: Union[Dataset, DataLoader],
        checkpoints: Union[str, List[str], Iterator],
        checkpoints_load_func: Callable = _load_flexible_state_dict,
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        vectorize: bool = False,
        nearest_neighbors: Optional[NearestNeighbors] = None,
        projection_dim: int = None,
        seed: int = 0,
    ) -> None:
        r"""
        A version of TracInCPFast which is optimized for "interactive" calls to
        `influence` for the purpose of calculating proponents / opponents, or
        influence scores. "Interactive" means there will be multiple calls to
        `influence`, with each call for a different batch of test examples, and
        subsequent calls rely on the results of previous calls. The implementation in
        this class has been optimized so that each call to `influence` is fast, so that
        it can be used for interactive analysis. This class should only be used for
        interactive use cases. It should not be used if `influence` will only be
        called once, because to enable fast calls to `influence`, time and memory
        intensive preprocessing is required in `__init__`. Furthermore, it should not
        be used to calculate self influencs scores - `TracInCPFast` should be used
        instead for that purpose. To enable interactive analysis, this implementation
        saves pre-computed vectors for all training examples in
        `influence_src_dataset`. Crucially, the influence score of a training
        example on a test example is simply the dot-product of their corresponding
        vectors, and proponents / opponents can be found by first storing vectors for
        training examples in a nearest-neighbor data structure, and then finding the
        nearest-neighbors for a test example in terms of dot-product (see appendix F
        of the TracIn paper). This class should only be used if calls to `influence`
        to obtain proponents / opponents or influence scores will be made in an
        "interactive" manner, and there is sufficient memory to store vectors for the
        entire `influence_src_dataset`. This is because in order to enable interactive
        analysis, this implementation incures overhead in ``__init__` to setup the
        nearest-neighbors data structure, which is both time and memory intensive, as
        vectors corresponding to all training examples needed to be stored. To reduce
        memory usage, this implementation enables random projections of those vectors.
        Note that the influence scores computed with random projections are less
        accurate, though correct in expectation.

        Args:
            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            final_fc_layer (torch.nn.Module or str): The last fully connected layer in
                    the network for which gradients will be approximated via fast random
                    projection method. Can be either the layer module itself, or the
                    fully qualified name of the layer if it is a defined attribute of
                    the passed `model`.
            influence_src_dataset (torch.utils.data.Dataset or torch.utils.DataLoader):
                    In the `influence` method, we either compute the influence score of
                    training examples on examples in a test batch, or self influence
                    scores for those training examples, depending on which mode is used.
                    This argument represents the training dataset containing those
                    training examples. In order to compute those influence scores, we
                    will create a Pytorch DataLoader yielding batches of training
                    examples that is then used for processing. If this argument is
                    already a Pytorch Dataloader, that DataLoader can be directly
                    used for processing. If it is instead a Pytorch Dataset, we will
                    create a DataLoader using it, with batch size specified by
                    `batch_size`. For efficiency purposes, the batch size of the
                    DataLoader used for processing should be as large as possible, but
                    not too large, so that certain intermediate quantities created
                    from a batch still fit in memory. Therefore, if
                    `influence_src_dataset` is a Dataset, `batch_size` should be large.
                    If `influence_src_dataset` was already a DataLoader to begin with,
                    it should have been constructed to have a large batch size.
            checkpoints (str or List of str or Iterator): Either the directory of the
                    path to store and retrieve model checkpoints, a list of
                    filepaths with checkpoints from which to load, or an iterator which
                    returns objects from which to load checkpoints.
            checkpoints_load_func (Callable, optional): The function to load a saved
                    checkpoint into a model to update its parameters, and get the
                    learning rate if it is saved. By default uses a utility to load a
                    model saved as a state dict.
                    Default: _load_flexible_state_dict
            loss_fn (Callable, optional): The loss function applied to model. `loss_fn`
                    must be a "reduction" loss function that reduces the per-example
                    losses in a batch, and returns a single scalar Tensor. Furthermore,
                    the reduction must be the *sum* of the per-example losses. For
                    instance, `nn.BCELoss(reduction="sum")` is acceptable, but
                    `nn.BCELoss(reduction="mean")` is *not* acceptable.
                    Default: None
            batch_size (int or None, optional): Batch size of the DataLoader created to
                    iterate through `influence_src_dataset`, if it is a Dataset.
                    `batch_size` should be chosen as large as possible so that certain
                    intermediate quantities created from a batch still fit in memory.
                    Specific implementations of `TracInCPBase` will detail the size of
                    the intermediate quantities. `batch_size` must be an int if
                    `influence_src_dataset` is a Dataset. If `influence_src_dataset`
                    is a DataLoader, then `batch_size` is ignored as an argument.
                    Default: 1
            vectorize (bool): Flag to use experimental vectorize functionality
                    for `torch.autograd.functional.jacobian`.
                    Default: False
            nearest_neighbors (NearestNeighbors, optional): The NearestNeighbors
                    instance for finding nearest neighbors. If None, defaults to
                    `AnnoyNearestNeighbors(n_trees=10)`.
                    Default: None
            projection_dim (int, optional): Each example will be represented in
                    the nearest neighbors data structure with a vector. This vector
                    is the concatenation of several "checkpoint vectors", each of which
                    is computed using a different checkpoint in the `checkpoints`
                    argument. If `projection_dim` is an int, it represents the
                    dimension we will project each "checkpoint vector" to, so that the
                    vector for each example will be of dimension at most
                    `projection_dim` * C, where C is the number of checkpoints.
                    Regarding the dimension of each vector, D: Let I be the dimension
                    of the output of the last fully-connected layer times the dimension
                    of the input of the last fully-connected layer. If `projection_dim`
                    is not `None`, then D = min(I * C, `projection_dim` * C).
                    Otherwise, D = I * C. In summary, if `projection_dim` is None, the
                    dimension of this vector will be determined by the size of the
                    input and output of the last fully-connected layer of `model`, and
                    the number of checkpoints. Otherwise, `projection_dim` must be an
                    int, and random projection will be performed to ensure that the
                    vector is of dimension no more than `projection_dim` * C.
                    `projection_dim` corresponds to the variable d in the top of page
                    15 of the TracIn paper: https://arxiv.org/pdf/2002.08484.pdf.
                    Default: None
            seed (int, optional): Because this implementation chooses a random
                    projection, its output is random. Setting this seed specifies the
                    random seed when choosing the random projection.
                    Default: 0
        """

        TracInCPFast.__init__(
            self,
            model,
            final_fc_layer,
            influence_src_dataset,
            checkpoints,
            checkpoints_load_func,
            loss_fn,
            batch_size,
            vectorize,
        )

        warnings.warn(
            (
                "WARNING: Using this implementation stores quantities related to the "
                "entire `influence_src_dataset` in memory, and may results in running "
                "out of memory. If this happens, consider using %s instead, for which "
                "each call to `influence` to compute influence scores or proponents "
                "will be slower, but may avoid running out of memory."
            )
            % "`TracInCPFast`"
        )

        self.nearest_neighbors = (
            AnnoyNearestNeighbors() if nearest_neighbors is None else nearest_neighbors
        )

        self.projection_dim = projection_dim

        torch.manual_seed(seed)  # for reproducibility
        self.projection_quantities = self._set_projections_tracincp_fast_rand_proj(
            self.influence_src_dataloader,
        )

        self.src_intermediate_quantities = (
            self._get_intermediate_quantities_tracincp_fast_rand_proj(
                self.influence_src_dataloader,
                self.projection_quantities,
            )
        )

        self._process_src_intermediate_quantities_tracincp_fast_rand_proj(
            self.src_intermediate_quantities,
        )

    def _influence(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
    ) -> Tensor:
        r"""
        Args:
            inputs (tuple of Any): A batch of examples. Does not represent labels,
                    which are passed as `targets`. The assumption is that
                    `self.model(*inputs)` produces the predictions for the batch.
            targets (tensor): The labels corresponding to the batch `inputs`. This
                    method is designed to be applied for a loss function, so labels
                    are required.

        Returns:
            influence_scores (tensor): Influence scores from the
            TracInCPFastRandProj method. Its shape is
            `(input_size, influence_src_dataset_size)`, where `input_size` is the
            number of examples in the test batch, and `influence_src_dataset_size` is
            the number of examples in training dataset `influence_src_dataset`. For
            example, `influence_scores[i][j]` is the influence score for the j-th
            training example to the i-th input example.
        """
        inputs_batch = (*inputs, targets)
        input_projections = self._get_intermediate_quantities_tracincp_fast_rand_proj(
            DataLoader(
                _DatasetFromList([inputs_batch]), shuffle=False, batch_size=None
            ),
            self.projection_quantities,
        )

        src_projections = self.src_intermediate_quantities

        return torch.matmul(input_projections, src_projections.T)

    def _get_k_most_influential(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
        k: int = 5,
        proponents: bool = True,
    ) -> KMostInfluentialResults:
        r"""
        Args:
            inputs (Tuple of Any): A tuple that represents a batch of examples. It does
                    not represent labels, which are passed as `targets`.
            targets (tensor): The labels corresponding to the batch `inputs`. This
                    method is designed to be applied for a loss function, so labels
                    are required.
            k (int, optional): The number of proponents or opponents to return per test
                    example.
                    Default: 5
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`)
                    Default: True

        Returns:
            (indices, influence_scores) (namedtuple): `indices` is a torch.long Tensor
                    that contains the indices of the proponents (or opponents) for each
                    test example. Its dimension is `(inputs_batch_size, k)`, where
                    `inputs_batch_size` is the number of examples in `inputs`. For
                    example, if `proponents==True`, `indices[i][j]` is the index of the
                    example in training dataset `influence_src_dataset` with the
                    k-th highest influence score for the j-th example in `inputs`.
                    `indices` is a `torch.long` tensor so that it can directly be used
                    to index other tensors. Each row of `influence_scores` contains the
                    influence scores for a different test example, in sorted order. In
                    particular, `influence_scores[i][j]` is the influence score of
                    example `indices[i][j]` in training dataset `influence_src_dataset`
                    on example `i` in the test batch represented by `inputs` and
                    `targets`.
        """
        inputs_batch = (*inputs, targets)
        input_projections = self._get_intermediate_quantities_tracincp_fast_rand_proj(
            DataLoader(
                _DatasetFromList([inputs_batch]), shuffle=False, batch_size=None
            ),
            self.projection_quantities,
        )
        multiplier = 1 if proponents else -1

        input_projections *= multiplier

        indices, distances = self.nearest_neighbors.get_nearest_neighbors(
            input_projections, k
        )

        distances *= multiplier

        return KMostInfluentialResults(indices, distances)

    def _self_influence(self):
        """
        NOT IMPLEMENTED - no need to implement `TracInCPFastRandProj._self_influence`,
        as `TracInCPFast._self_influence` is sufficient - the latter does not benefit
        from random projections, since no quantities associated with a training
        example are stored (other than its self influence score)

        Returns:
            self influence scores (Tensor): 1-d Tensor containing self influence
                    scores for all examples in training dataset
                    `influence_src_dataset`.
        """
        warnings.warn(
            (
                "WARNING: If calculating self influence scores, when only considering "
                "gradients with respect to the last fully-connected layer, "
                "`TracInCPFastRandProj` should not be used. Instead, please use "
                "`TracInCPFast`. This is because when calculating self influence "
                "scores, no quantities associated with a training example are stored "
                "so that memory-saving benefit of the random projections used by "
                "`TracInCPFastRandProj`needed. Further considering the fact that "
                "random projections results only in approximate self influence "
                "scores, there is no reason to use `TracInCPFastRandProj` when "
                "calculating self-influence scores."
            )
        )
        raise NotImplementedError

    def influence(  # type: ignore[override]
        self,
        inputs: Any,
        targets: Tensor,
        k: int = 5,
        proponents: bool = True,
        unpack_inputs: bool = True,
    ) -> Union[Tensor, KMostInfluentialResults]:
        r"""
        This is the key method of this class, and can be run in 2 different modes,
        where the mode that is run depends on the arguments passed to this method

        - influence score mode: This mode is used if `inputs` is not None, and `k` is
          None. This mode computes the influence score of every example in
          training dataset `influence_src_dataset` on every example in the test
          batch represented by `inputs` and `targets`.

        - k-most influential mode: This mode is used if `inputs` is not None, and
          `k` is not None, and an int. This mode computes the proponents or
          opponents of every example in the test batch represented by `inputs`
          and `targets`. In particular, for each test example in the test batch,
          this mode computes its proponents (resp. opponents), which are the
          indices in the training dataset `influence_src_dataset` of the training
          examples with the `k` highest (resp. lowest) influence scores on the
          test example. Proponents are computed if `proponents` is True.
          Otherwise, opponents are computed. For each test example, this method
          also returns the actual influence score of each proponent (resp.
          opponent) on the test example.

        Note that unlike `TracInCPFast`, this class should *not* be run in self
        influence mode.  To compute self influence scores when only considering
        gradients in the last fully-connected layer, please use `TracInCPFast` instead.

        Args:
            inputs (Any, optional): If not provided or `None`, the self influence mode
                    will be run. Otherwise, `inputs` is the test batch that will be
                    used when running in either influence score or k-most influential
                    mode. If the argument `unpack_inputs` is False, the
                    assumption is that `self.model(inputs)` produces the predictions
                    for a batch, and `inputs` can be of any type. Otherwise if the
                    argument `unpack_inputs` is True, the assumption is that
                    `self.model(*inputs)` produces the predictions for a batch, and
                    `inputs` will need to be a tuple. In other words, `inputs` will be
                    unpacked as an argument when passing to `self.model`.
                    Default: None
            targets (tensor): The labels corresponding to the batch `inputs`. This
                    method is designed to be applied for a loss function, so labels
                    are required, unless running in "self influence" mode.
            k (int, optional): If not provided or `None`, the influence score mode will
                    be run. Otherwise, the k-most influential mode will be run,
                    and `k` is the number of proponents / opponents to return per
                    example in the test batch.
                    Default: None
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`), if running in k-most influential
                    mode.
                    Default: True
            unpack_inputs (bool, optional): Whether to unpack the `inputs` argument to
                    when passing it to `model`, if `inputs` is a tuple (no unpacking
                    done otherwise).
                    Default: True

        Returns:

            The return value of this method depends on which mode is run

            - influence score mode: if this mode is run (`inputs is not None, `k` is
              None), returns a 2D tensor `influence_scores` of shape
              `(input_size, influence_src_dataset_size)`, where `input_size` is
              the number of examples in the test batch, and
              `influence_src_dataset_size` is the number of examples in
              training dataset `influence_src_dataset`. In other words,
              `influence_scores[i][j]` is the influence score of the `j`-th
              example in `influence_src_dataset` on the `i`-th example in the
              test batch.
            - most influential mode: if this mode is run (`inputs` is not None,
              `k` is an int), returns `indices`, which is a 2D tensor of shape
              `(input_size, k)`, where `input_size` is the number of examples
              in the test batch. If computing proponents (resp. opponents),
              `indices[i][j]` is the index in training dataset
              `influence_src_dataset` of the example with the `j`-th highest
              (resp. lowest) influence score (out of the examples in
              `influence_src_dataset`) on the `i`-th example in the test batch.
        """
        msg = (
            "Since `inputs` is None, this suggests `TracInCPFastRandProj` is being "
            "used in self influence mode. However, `TracInCPFastRandProj` should not "
            "be used to compute self influence scores.  If desiring self influence "
            "scores which only consider gradients in the last fully-connected layer, "
            "please use `TracInCPFast` instead."
        )
        assert inputs is not None, msg
        return TracInCPBase.influence(
            self, inputs, targets, k, proponents, unpack_inputs
        )

    def _set_projections_tracincp_fast_rand_proj(
        self,
        dataloader: DataLoader,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        returns the variables `jacobian_projection` and `layer_input_projection`
        if needed, based on `self.projection_dim`. The two variables are
        used by `self._get_intermediate_quantities_fast_rand_proj`. They are both None
        if projection is not needed, due to the intermediate quantities (see the
        `_get_intermediate_quantities_fast_rand_proj` method for details) being no
        greater than `self.projection_dim` * C even without projection, where C is the
        number of checkpoints in the `checkpoints` argument to
        `TracInCPFastRandProj.__init__`.

        Args:
            dataloader (DataLoader): determining the projection requires knowing the
                    dimensionality of the last layer's parameters (`jacobian_dim`
                    below) and its input (`layer_input_dim` below). These are
                    determined by passing a batch to `self.model`. `dataloader`
                    provides that batch.

        Returns:
            jacobian_projection (tensor or None): Projection matrix to apply to
                    Jacobian of last layer to reduce its dimension, if needed.
                    None otherwise.
            input_projection (tensor or None): Projection matrix to apply to input of
                    last layer to reduce its dimension, if needed. None otherwise.
        """
        # figure out projection dimensions, if needed

        projection_dim = self.projection_dim

        projection_quantities = None

        if not (projection_dim is None):

            # figure out original dimensions by looking at data, passing through network
            self.checkpoints_load_func(self.model, next(iter(self.checkpoints)))

            batch = next(iter(dataloader))
            batch_jacobians, batch_layer_inputs = _basic_computation_tracincp_fast(
                self,
                batch[0:-1],
                batch[-1],
            )

            jacobian_dim = batch_jacobians.shape[
                1
            ]  # this is the dimension of the output of the last fully-connected layer
            layer_input_dim = batch_layer_inputs.shape[
                1
            ]  # this is the dimension of the input of the last fully-connected layer

            # choose projection if needed
            # without projection, the dimension of the intermediate quantities returned
            # by `_get_intermediate_quantities_fast_rand_proj` will be
            # `jacobian_dim` * `layer_input_dim` * number of checkpoints
            # this is because for each checkpoint, we compute a "partial" intermediate
            # quantity, and the intermediate quantity is the concatenation of the
            # "partial" intermediate quantities, and the dimension of each "partial"
            # intermediate quantity, without projection, is `jacobian_dim` *
            # `layer_input_dim`. However, `projection_dim` refers to the maximum
            # allowable dimension of the "partial" intermediate quantity. Therefore,
            # we only project if `jacobian_dim` * `layer_input_dim` > `projection_dim`.
            # `projection_dim` corresponds to the variable d in the top of page 15 of
            # the TracIn paper: https://arxiv.org/pdf/2002.08484.pdf.
            if jacobian_dim * layer_input_dim > projection_dim:
                jacobian_projection_dim = min(int(projection_dim ** 0.5), jacobian_dim)
                layer_input_projection_dim = min(
                    int(projection_dim ** 0.5), layer_input_dim
                )
                jacobian_projection = torch.normal(
                    torch.zeros(jacobian_dim, jacobian_projection_dim),
                    1.0 / jacobian_projection_dim ** 0.5,
                )
                layer_input_projection = torch.normal(
                    torch.zeros(layer_input_dim, layer_input_projection_dim),
                    1.0 / layer_input_projection_dim ** 0.5,
                )

                projection_quantities = jacobian_projection, layer_input_projection

        return projection_quantities

    def _process_src_intermediate_quantities_tracincp_fast_rand_proj(
        self,
        src_intermediate_quantities: torch.Tensor,
    ):
        """
        Assumes `self._get_intermediate_quantities_tracin_fast_rand_proj` returns
        vector representations for each example, and that influence between a
        training and test example is obtained by taking the dot product of their
        vector representations. In this case, given a test example, its proponents
        can be found by storing the vector representations for training examples
        into a data structure enablng fast largest-dot-product computation. This
        method creates that data structure. This method has side effects.

        Args:
            src_intermediate_quantities (tensor): the output of the
                    `_get_intermediate_quantities_tracin_fast_rand_proj` function when
                    applied to training dataset `influence_src_dataset`. This
                    output is the vector representation of all training examples.
                    The dot product between the representation of a training example
                    and the representation of a test example gives the influence score
                    of the training example on the test example.
        """
        self.nearest_neighbors.setup(src_intermediate_quantities)

    def _get_intermediate_quantities_tracincp_fast_rand_proj(
        self,
        dataloader: DataLoader,
        projection_quantities: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        r"""
        This method computes vectors that can be used to compute influence. (see
        Appendix F, page 15). Crucially, the influence score between a test example
        and a training example is simply the dot product of their respective
        vectors. This means that the training example with the largest influence score
        on a given test example can be found using a nearest-neighbor (more
        specifically, largest dot-product) data structure.

        Args:
            dataloader (DataLoader): DataLoader for which the intermediate quantities
                    are computed.
            projection_quantities (tuple or None): Is either the two tensors defining
                    the randomized projections to apply, or None, which means no
                    projection is to be applied.

        Returns:
            checkpoint_projections (tensor): A tensor of dimension
                    (N, D * C), where N is total number of examples in `dataloader`, C
                    is the number of checkpoints passed as the `checkpoints` argument
                    of `TracInCPFastRandProj.__init__`, and each row represents the
                    vector for an example. Regarding D: Let I be the dimension of the
                    output of the last fully-connected layer times the dimension of the
                    input of the last fully-connected layer. If `self.projection_dim`
                    is specified in initialization,
                    D = min(I * C, `self.projection_dim` * C). Otherwise, D = I * C.
                    In summary, if `self.projection_dim` is None, the dimension of each
                    vector will be determined by the size of the input and output of
                    the last fully-connected layer of `model`. Otherwise,
                    `self.projection_dim` must be an int, and random projection will be
                    performed to ensure that the vector is of dimension no more than
                    `self.projection_dim` * C. `self.projection_dim` corresponds to
                    the variable d in the top of page 15 of the TracIn paper:
                    https://arxiv.org/pdf/2002.08484.pdf.
        """
        checkpoint_projections: List[Any] = [[] for _ in self.checkpoints]

        if projection_quantities is None:
            project = False
        else:
            project = True
            jacobian_projection, layer_input_projection = projection_quantities

        for (j, checkpoint) in enumerate(self.checkpoints):
            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)
            learning_rate_root = learning_rate ** 0.5

            for batch in dataloader:

                batch_jacobians, batch_layer_inputs = _basic_computation_tracincp_fast(
                    self,
                    batch[0:-1],
                    batch[-1],
                )

                if project:

                    batch_jacobians = torch.matmul(batch_jacobians, jacobian_projection)

                    batch_layer_inputs = torch.matmul(
                        batch_layer_inputs, layer_input_projection
                    )

                checkpoint_projections[j].append(
                    torch.matmul(
                        torch.unsqueeze(batch_jacobians, 2),
                        torch.unsqueeze(batch_layer_inputs, 1),
                    ).flatten(start_dim=1)
                    * learning_rate_root
                )

            checkpoint_projections[j] = torch.cat(checkpoint_projections[j], dim=0)

        return torch.cat(checkpoint_projections, dim=1)
