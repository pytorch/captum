import tempfile
from typing import Callable

import torch

import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from parameterized import parameterized
from tests.helpers.basic import assertTensorAlmostEqual, BaseTest
from tests.influence._utils.common import (
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
)
from torch.utils.data import DataLoader


class TestTracInIntermediateQuantities(BaseTest):
    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("none", DataInfluenceConstructor(TracInCP)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_aggregate(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        """
        tests that calling `compute_intermediate_quantities` with `aggregate=True`
        does give the same result as calling it with `aggregate=False`, and then
        summing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (net, train_dataset,) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                return_test_data=False,
            )

            # create a dataloader that yields batches from the dataset
            train_dataset = DataLoader(train_dataset, batch_size=5)

            # create tracin instance
            criterion = nn.MSELoss(reduction=reduction)
            batch_size = 5

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            intermediate_quantities = tracin.compute_intermediate_quantities(
                train_dataset, aggregate=False
            )
            aggregated_intermediate_quantities = tracin.compute_intermediate_quantities(
                train_dataset, aggregate=True
            )

            assertTensorAlmostEqual(
                self,
                torch.sum(intermediate_quantities, dim=0, keepdim=True),
                aggregated_intermediate_quantities,
                delta=1e-4,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )

    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
                ("none", DataInfluenceConstructor(TracInCP)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_api(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        """
        tests that the result of calling the public method
        `compute_intermediate_quantities` for a DataLoader of batches is the same as
        when the batches are collated into a single batch
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (net, train_dataset,) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                return_test_data=False,
            )

            # create a single batch representing the entire dataset
            single_batch = next(
                iter(DataLoader(train_dataset, batch_size=len(train_dataset)))
            )

            # create a dataloader that yields batches from the dataset
            dataloader = DataLoader(train_dataset, batch_size=5)

            # create tracin instance
            criterion = nn.MSELoss(reduction=reduction)
            batch_size = 5
            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # compute intermediate quantities using `compute_intermediate_quantities`
            # when passing in a single batch
            single_batch_intermediate_quantities = (
                tracin.compute_intermediate_quantities(single_batch)
            )

            # compute intermediate quantities using `compute_intermediate_quantities`
            # when passing in a dataloader with the same examples
            dataloader_intermediate_quantities = tracin.compute_intermediate_quantities(
                dataloader,
            )

            # the two self influences should be equal
            assertTensorAlmostEqual(
                self,
                single_batch_intermediate_quantities,
                dataloader_intermediate_quantities,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )

    @parameterized.expand(
        [
            (
                reduction,
                constructor,
                intermediate_quantities_tracin_constructor,
                unpack_inputs,
            )
            for unpack_inputs in [True, False]
            for (
                reduction,
                constructor,
                intermediate_quantities_tracin_constructor,
            ) in [
                (
                    "sum",
                    DataInfluenceConstructor(TracInCPFast),
                    DataInfluenceConstructor(TracInCPFastRandProj),
                ),
                (
                    "none",
                    DataInfluenceConstructor(TracInCP),
                    DataInfluenceConstructor(TracInCP),
                ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_consistent(
        self,
        reduction: str,
        tracin_constructor: Callable,
        intermediate_quantities_tracin_constructor: Callable,
        unpack_inputs: bool,
    ) -> None:
        """
        Since the influence score of a test batch on a training data should be the dot
        product of their intermediate quantities, checks that this is the case, by
        computing the influence score 2 different ways and checking they give the same
        results: 1) with the `influence` method, and by using the
        `compute_intermediate_quantities` method on the test and training data, and
        taking the dot product. No projection should be done.  Otherwise, the
        projection will cause error. For 1), we use an implementation that does not use
        intermediate quantities, i.e. `TracInCPFast`.  For 2), we use a method that
        does use intermediate quantities, i.e. `TracInCPFastRandProj`. Since the
        methods for the 2 cases are different, we need to parametrize the test with 2
        different tracin constructors. `tracin_constructor` is the constructor for the
        tracin implementation for case 1.  `intermediate_quantities_tracin_constructor`
        is the constructor for the tracin implementation for case 2.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_features,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True)

            # create a dataloader that yields batches from the dataset
            train_dataset = DataLoader(train_dataset, batch_size=5)

            # create tracin instance
            criterion = nn.MSELoss(reduction=reduction)
            batch_size = 5

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # create tracin instance which exposes `intermediate_quantities`
            intermediate_quantities_tracin = intermediate_quantities_tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # compute influence scores without using `compute_intermediate_quantities`
            scores = tracin.influence(
                test_features, test_labels, unpack_inputs=unpack_inputs
            )

            # compute influence scores using `compute_intermediate_quantities`
            # we combine `test_features` and `test_labels` into a single tuple
            # `test_batch` to pass to the model, with the assumption that
            # `model(test_batch[0:-1]` produces the predictions, and `test_batch[-1]`
            # are the labels.  We do this due to the assumptions made by the
            # `compute_intermediate_quantities` method. Therefore, how we
            # form `test_batch` depends on whether `unpack_inputs` is True or False
            if not unpack_inputs:
                # `test_features` is a Tensor
                test_batch = (test_features, test_labels)
            else:
                # `test_features` is a tuple, so we unpack it to place in tuple,
                # along with `test_labels`
                test_batch = (*test_features, test_labels)  # type: ignore[assignment]

            # the influence score is the dot product of intermediate quantities
            intermediate_quantities_scores = torch.matmul(
                intermediate_quantities_tracin.compute_intermediate_quantities(
                    test_batch
                ),
                intermediate_quantities_tracin.compute_intermediate_quantities(
                    train_dataset
                ).T,
            )

            # the scores computed using the two methods should be the same
            assertTensorAlmostEqual(
                self,
                scores,
                intermediate_quantities_scores,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )
