import tempfile
import unittest
from typing import Callable, cast

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
    is_gpu_ready,
)
class TestTracInGetKMostInfluential(BaseTest):
    """
    This test constructs a random BasicLinearNet, and checks that the proponents
    obtained by calling `influence` and sorting are equal to the proponents
    obtained by calling `_k_most_influential`.  Those calls are made through
    the calls to wrapper method `influence`.
    """

    @parameterized.expand(
        [
            (reduction, constr, unpack_inputs, proponents, batch_size, k, use_gpu)
            # calls test helper method `test_tracin_k_most_influential` for several
            # combinations of `batch_size` and `k`.  This is important because the
            # behavior of `_k_most_influential` depends on whether `k` is larger
            # than `batch_size`.
            for (batch_size, k) in [(4, 7), (7, 4), (40, 5), (5, 40), (40, 45)]
            for unpack_inputs in [True, False]
            for proponents in [True, False]
            for use_gpu in [True, False]
            for reduction, constr in [
                ("none", DataInfluenceConstructor(TracInCP)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCPFastRandProjTests",
                        sample_wise_grads_per_batch=True,
                    ),
                ),
                ("sum", DataInfluenceConstructor(TracInCPFast)),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
                ("mean", DataInfluenceConstructor(TracInCPFast)),
                ("mean", DataInfluenceConstructor(TracInCPFastRandProj)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_k_most_influential(
        self,
        reduction: str,
        tracin_constructor: Callable,
        unpack_inputs: bool,
        proponents: bool,
        batch_size: int,
        k: int,
        use_gpu: bool,
    ) -> None:

        is_gpu_ready_ = is_gpu_ready(
            use_gpu,
            tracin_constructor=cast(DataInfluenceConstructor, tracin_constructor),
        )
        if not is_gpu_ready_ and use_gpu:
            raise unittest.SkipTest(
                "GPU test is skipped because GPU device is unavailable or \
                 `sample_wise_trick` option is used."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                True,
                is_gpu_ready_,
            )

            self.assertTrue(isinstance(reduction, str))
            self.assertTrue(callable(tracin_constructor))

            criterion = nn.MSELoss(reduction=reduction)

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            train_scores = tracin.influence(
                test_samples, test_labels, k=None, unpack_inputs=unpack_inputs
            )
            sort_idx = torch.argsort(train_scores, dim=1, descending=proponents)[:, 0:k]
            idx, _train_scores = tracin.influence(
                test_samples,
                test_labels,
                k=k,
                proponents=proponents,
                unpack_inputs=unpack_inputs,
            )

            for i in range(len(idx)):
                # check that idx[i] is correct
                assertTensorAlmostEqual(
                    self,
                    train_scores[i, idx[i]],
                    train_scores[i, sort_idx[i]],
                    delta=0.0,
                    mode="max",
                )
                # check that _train_scores[i] is correct
                assertTensorAlmostEqual(
                    self,
                    _train_scores[i],
                    train_scores[i, sort_idx[i]],
                    delta=0.001,
                    mode="max",
                )