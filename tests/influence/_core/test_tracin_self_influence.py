import tempfile
import unittest
from typing import Callable, cast

import torch
import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import TracInCPFast
from parameterized import parameterized
from tests.helpers.basic import assertTensorAlmostEqual, BaseTest
from tests.influence._utils.common import (
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
    is_gpu_ready,
)
from torch.utils.data import DataLoader


class TestTracInSelfInfluence(BaseTest):
    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs, use_gpu)
            for unpack_inputs in [True, False]
            for use_gpu in [True, False]
            for (reduction, constructor) in [
                ("none", DataInfluenceConstructor(TracInCP)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCP_sample_wise_grads_per_batch",
                        sample_wise_grads_per_batch=True,
                    ),
                ),
                ("sum", DataInfluenceConstructor(TracInCPFast)),
                ("mean", DataInfluenceConstructor(TracInCPFast)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_self_influence(
        self,
        reduction: str,
        tracin_constructor: Callable,
        unpack_inputs: bool,
        use_gpu: bool,
    ) -> None:
        is_gpu_ready_ = is_gpu_ready(
            use_gpu,
            tracin_constructor=cast(DataInfluenceConstructor, tracin_constructor),
        )
        if not is_gpu_ready_ and use_gpu:
            raise unittest.SkipTest(
                "GPU test is skipped because GPU device is unavailable."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            (net, train_dataset,) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                False,
                is_gpu_ready_,
            )

            # compute tracin_scores of training data on training data
            criterion = nn.MSELoss(reduction=reduction)
            batch_size = 5

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )
            train_scores = tracin.influence(
                train_dataset.samples,
                train_dataset.labels,
                k=None,
                unpack_inputs=unpack_inputs,
            )

            # calculate self_tracin_scores
            self_tracin_scores = tracin.self_influence(
                DataLoader(train_dataset, batch_size=batch_size),
                outer_loop_by_checkpoints=False,
            )

            # check that self_tracin scores equals the diagonal of influence scores
            assertTensorAlmostEqual(
                self,
                torch.diagonal(train_scores),
                self_tracin_scores,
                delta=0.01,
                mode="max",
            )

            # check that setting `outer_loop_by_checkpoints=False` and
            # `outer_loop_by_checkpoints=True` gives the same self influence scores
            self_tracin_scores_by_checkpoints = tracin.self_influence(
                DataLoader(train_dataset, batch_size=batch_size),
                outer_loop_by_checkpoints=True,
            )
            assertTensorAlmostEqual(
                self,
                self_tracin_scores_by_checkpoints,
                self_tracin_scores,
                delta=0.01,
                mode="max",
            )

    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("none", DataInfluenceConstructor(TracInCP)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCP,
                        sample_wise_grads_per_batch=True,
                    ),
                ),
                ("sum", DataInfluenceConstructor(TracInCPFast)),
                ("mean", DataInfluenceConstructor(TracInCPFast)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_self_influence_dataloader_vs_single_batch(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        # tests that the result of calling the public method `self_influence` for a
        # DataLoader of batches is the same as when the batches are collated into a
        # single batch
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=False)

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

            # compute self influence using `self_influence` when passing in a single
            # batch
            single_batch_self_influence = tracin.self_influence(single_batch)

            # compute self influence using `self_influence` when passing in a
            # dataloader with the same examples
            dataloader_self_influence = tracin.self_influence(dataloader)

            # the two self influences should be equal
            assertTensorAlmostEqual(
                self,
                single_batch_self_influence,
                dataloader_self_influence,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )
