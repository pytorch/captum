import tempfile
from typing import Callable, Union

import torch
import torch.nn as nn
from captum.influence._core.tracincp import TracInCP

from parameterized import parameterized
from tests.helpers import BaseTest
from tests.helpers.basic import assertTensorAlmostEqual
from tests.influence._utils.common import (
    _format_batch_into_tuple,
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
)


class TestTracInGetKMostInfluential(BaseTest):

    use_gpu_list = (
        [False, "cuda", "cuda_data_parallel"]
        if torch.cuda.is_available() and torch.cuda.device_count() != 0
        else [False]
    )

    param_list = []
    for batch_size, k in [(4, 7), (7, 4), (40, 5), (5, 40), (40, 45)]:
        for unpack_inputs in [True, False]:
            for proponents in [True, False]:
                for use_gpu in use_gpu_list:
                    for reduction, constr, aggregate in [
                        (
                            "none",
                            DataInfluenceConstructor(
                                TracInCP, name="TracInCP_all_layers"
                            ),
                            False,
                        ),
                        (
                            "none",
                            DataInfluenceConstructor(
                                TracInCP, name="TracInCP_all_layers"
                            ),
                            True,
                        ),
                        (
                            "none",
                            DataInfluenceConstructor(
                                TracInCP,
                                name="linear2",
                                layers=(
                                    ["module.linear2"]
                                    if use_gpu == "cuda_data_parallel"
                                    else ["linear2"]
                                ),
                            ),
                            False,
                        ),
                    ]:
                        if not (
                            "sample_wise_grads_per_batch" in constr.kwargs
                            and constr.kwargs["sample_wise_grads_per_batch"]
                            and use_gpu
                        ):
                            param_list.append(
                                (
                                    reduction,
                                    constr,
                                    unpack_inputs,
                                    proponents,
                                    batch_size,
                                    k,
                                    use_gpu,
                                    aggregate,
                                )
                            )

    @parameterized.expand(
        param_list,
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
        use_gpu: Union[bool, str],
        aggregate: bool,
    ) -> None:
        """
        This test constructs a random BasicLinearNet, and checks that the proponents
        obtained by calling `influence` and sorting are equal to the proponents
        obtained by calling `_k_most_influential`.  Those calls are made through
        the calls to wrapper method `influence`.
        """
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
                use_gpu,
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
                _format_batch_into_tuple(test_samples, test_labels, unpack_inputs),
                k=None,
                aggregate=aggregate,
            )
            sort_idx = torch.argsort(train_scores, dim=1, descending=proponents)[:, 0:k]
            idx, _train_scores = tracin.influence(
                _format_batch_into_tuple(test_samples, test_labels, unpack_inputs),
                k=k,
                proponents=proponents,
                aggregate=aggregate,
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
