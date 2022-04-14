import tempfile
from typing import Callable

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


class TestTracInDataLoader(BaseTest):
    """
    This tests that the influence score computed when a Dataset is fed to the
    `self.tracin_constructor` and when a DataLoader constructed using the same
    Dataset is fed to `self.tracin_constructor` gives the same results.
    """

    @parameterized.expand(
        [
            (
                reduction,
                constr,
                unpack_inputs,
            )
            for unpack_inputs in [False, True]
            for reduction, constr in [
                ("none", DataInfluenceConstructor(TracInCP)),
                ("sum", DataInfluenceConstructor(TracInCPFast)),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj,
                        name="TracInCPFastRandProj_1DProj",
                        projection_dim=1,
                    ),
                ),
            ]
        ],
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    def test_tracin_dataloader(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:

        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 5

            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True)

            self.assertTrue(isinstance(reduction, str))
            criterion = nn.MSELoss(reduction=reduction)

            self.assertTrue(callable(tracin_constructor))
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

            tracin_dataloader = tracin_constructor(
                net,
                DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
                tmpdir,
                None,
                criterion,
            )

            train_scores_dataloader = tracin_dataloader.influence(
                test_samples, test_labels, k=None, unpack_inputs=unpack_inputs
            )

            assertTensorAlmostEqual(
                self,
                train_scores,
                train_scores_dataloader,
                delta=0.0,
                mode="max",
            )
