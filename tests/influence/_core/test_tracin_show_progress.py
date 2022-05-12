import io
import tempfile
import unittest
import unittest.mock
from typing import Callable

import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
)
from parameterized import parameterized
from tests.helpers.basic import BaseTest
from tests.influence._utils.common import (
    get_random_model_and_data,
    DataInfluenceConstructor,
    build_test_name_func,
)


class TestTracInShowProgress(BaseTest):
    """
    This tests that the progress bar correctly shows a "100%" message at some point in
    the relevant computations.  Progress bars are shown for calls to the `influence`
    method for all 3 modes.  This is why 3 different modes are tested, and the mode
    being tested is a parameter in the test.  `TracInCPFastRandProj.influence` is not
    tested, because none of its modes involve computations over the entire training
    dataset, so that no progress bar is shown (the computation is instead done in
    `TracInCPFastRandProj.__init__`.  TODO: add progress bar for computations done
    in `TracInCPFastRandProj.__init__`).
    """

    @parameterized.expand(
        [
            (
                reduction,
                constr,
                mode,
            )
            for reduction, constr in [
                (
                    "none",
                    DataInfluenceConstructor(TracInCP),
                ),
                (
                    "sum",
                    DataInfluenceConstructor(TracInCPFast),
                ),
            ]
            for mode in ["self influence", "influence", "k-most"]
        ],
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_tracin_show_progress(
        self,
        reduction: str,
        tracin_constructor: Callable,
        mode: str,
        mock_stderr,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 5

            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(
                tmpdir, unpack_inputs=False, return_test_data=True
            )

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

            if mode == "self influence":
                tracin.influence(show_progress=True)
                output = mock_stderr.getvalue()
                self.assertTrue(
                    (
                        (
                            f"Using {tracin.get_name()} to compute self influence "
                            "for training batches: 100%"
                        )
                        in output
                    ),
                    f"Error progress output: {repr(output)}",
                )
            elif mode == "influence":

                tracin.influence(
                    test_samples,
                    test_labels,
                    k=None,
                    show_progress=True,
                )
                output = mock_stderr.getvalue()
                self.assertTrue(
                    (
                        (
                            f"Using {tracin.get_name()} to compute influence "
                            "for training batches: 100%"
                        )
                        in output
                    ),
                    f"Error progress output: {repr(output)}",
                )
            elif mode == "k-most":

                tracin.influence(
                    test_samples,
                    test_labels,
                    k=2,
                    proponents=True,
                    show_progress=True,
                )
                output = mock_stderr.getvalue()
                self.assertTrue(
                    (
                        (
                            f"Using {tracin.get_name()} to perform computation for "
                            "getting proponents. Processing training batches: 100%"
                        )
                        in output
                    ),
                    f"Error progress output: {repr(output)}",
                )
                mock_stderr.seek(0)
                mock_stderr.truncate(0)

                tracin.influence(
                    test_samples,
                    test_labels,
                    k=2,
                    proponents=False,
                    show_progress=True,
                )
                output = mock_stderr.getvalue()
                self.assertTrue(
                    (
                        (
                            f"Using {tracin.get_name()} to perform computation for "
                            "getting opponents. Processing training batches: 100%"
                        )
                        in output
                    ),
                    f"Error progress output: {repr(output)}",
                )
            else:
                raise Exception("unknown test mode")

            mock_stderr.seek(0)
            mock_stderr.truncate(0)
