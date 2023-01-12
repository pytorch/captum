import tempfile
from typing import Callable

import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import TracInCPFast

from parameterized import parameterized
from tests.helpers.basic import BaseTest
from tests.influence._utils.common import (
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
)


class TestTracinValidator(BaseTest):

    param_list = []
    for reduction, constr in [
        (
            "none",
            DataInfluenceConstructor(TracInCP, name="TracInCP"),
        ),
        (
            "mean",
            DataInfluenceConstructor(
                TracInCPFast,
                name="TracInCpFast",
            ),
        ),
    ]:
        param_list.append((reduction, constr))

    @parameterized.expand(
        param_list,
        name_func=build_test_name_func(),
    )
    def test_tracin_require_inputs_dataset(
        self,
        reduction,
        tracin_constructor: Callable,
    ) -> None:
        """
        This test verifies that tracinCP and tracinCPFast
        influence methods required `inputs_dataset`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs=False)

            criterion = nn.MSELoss(reduction=reduction)

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                loss_fn=criterion,
                batch_size=1,
            )
            with self.assertRaisesRegex(AssertionError, "required."):
                tracin.influence(None, k=None)
