import os
import tempfile
from typing import Callable, cast, Optional

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
    CoefficientNet,
    DataInfluenceConstructor,
    IdentityDataset,
    isSorted,
    RangeDataset,
)


class TestTracInRegression(BaseTest):
    def _test_tracin_regression_setup(self, tmpdir: str, features: int):
        low = 1
        high = 17
        dataset = RangeDataset(low, high, features)
        net = CoefficientNet(in_features=features)

        checkpoint_name = "-".join(["checkpoint-reg", "0" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        weights = [0.4379, 0.1653, 0.5132, 0.3651, 0.9992]

        for i, weight in enumerate(weights):
            net.fc1.weight.data.fill_(weight)
            checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
            torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        return dataset, net

    @parameterized.expand(
        [
            (reduction, constructor, mode, dim)
            for dim in [1, 20]
            for (mode, reduction, constructor) in [
                ("check_idx", "none", DataInfluenceConstructor(TracInCP)),
                ("sample_wise_trick", None, DataInfluenceConstructor(TracInCP)),
                ("check_idx", "sum", DataInfluenceConstructor(TracInCPFast)),
                ("check_idx", "sum", DataInfluenceConstructor(TracInCPFastRandProj)),
                ("check_idx", "mean", DataInfluenceConstructor(TracInCPFast)),
                ("check_idx", "mean", DataInfluenceConstructor(TracInCPFastRandProj)),
                (
                    "check_idx",
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj,
                        name="TracInCPFastRandProj1DimensionalProjection",
                        projection_dim=1,
                    ),
                ),
            ]
        ],
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    def test_tracin_regression(
        self,
        reduction: Optional[str],
        tracin_constructor: Callable,
        mode: str,
        features: int,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 4

            dataset, net = self._test_tracin_regression_setup(tmpdir, features)

            # check influence scores of training data

            train_inputs = dataset.samples
            train_labels = dataset.labels

            test_inputs = (
                torch.arange(17, 33, dtype=torch.float).unsqueeze(1).repeat(1, features)
            )
            test_labels = test_inputs

            self.assertTrue(callable(tracin_constructor))

            if mode == "check_idx":

                self.assertTrue(isinstance(reduction, str))
                criterion = nn.MSELoss(reduction=cast(str, reduction))

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )

                train_scores = tracin.influence(train_inputs, train_labels)
                idx, _ = tracin.influence(
                    train_inputs, train_labels, k=len(dataset), proponents=True
                )
                # check that top influence is one with maximal value
                # (and hence gradient)
                for i in range(len(idx)):
                    self.assertEqual(idx[i][0], 15)

                # check influence scores of test data
                test_scores = tracin.influence(test_inputs, test_labels)
                idx, _ = tracin.influence(
                    test_inputs, test_labels, k=len(test_inputs), proponents=True
                )
                # check that top influence is one with maximal value
                # (and hence gradient)
                for i in range(len(idx)):
                    self.assertTrue(isSorted(idx[i]))

            if mode == "sample_wise_trick":

                criterion = nn.MSELoss(reduction="none")

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=True,
                )

                train_scores = tracin.influence(train_inputs, train_labels)
                train_scores_sample_wise_trick = tracin_sample_wise_trick.influence(
                    train_inputs, train_labels
                )
                assertTensorAlmostEqual(
                    self, train_scores, train_scores_sample_wise_trick
                )

                test_scores = tracin.influence(test_inputs, test_labels)
                test_scores_sample_wise_trick = tracin_sample_wise_trick.influence(
                    test_inputs, test_labels
                )
                assertTensorAlmostEqual(
                    self, test_scores, test_scores_sample_wise_trick
                )

    @parameterized.expand(
        [
            (
                "sum",
                DataInfluenceConstructor(TracInCP, sample_wise_grads_per_batch=True),
            ),
            ("sum", DataInfluenceConstructor(TracInCPFast)),
            ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("mean", DataInfluenceConstructor(TracInCPFast)),
            ("mean", DataInfluenceConstructor(TracInCPFastRandProj)),
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_regression_1D_numerical(
        self, reduction: str, tracin_constructor: Callable
    ) -> None:

        low = 1
        high = 17
        features = 1
        dataset = RangeDataset(low, high, features)
        net = CoefficientNet()
        self.assertTrue(isinstance(reduction, str))
        criterion = nn.MSELoss(reduction=cast(str, reduction))
        batch_size = 4
        weights = [0.4379, 0.1653, 0.5132, 0.3651, 0.9992]

        train_inputs = dataset.samples
        train_labels = dataset.labels

        with tempfile.TemporaryDirectory() as tmpdir:

            for i, weight in enumerate(weights):
                net.fc1.weight.data.fill_(weight)
                checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
                torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

            self.assertTrue(callable(tracin_constructor))
            tracin = tracin_constructor(
                net,
                dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            train_scores = tracin.influence(train_inputs, train_labels, k=None)

            r"""
            Derivation for gradient / resulting TracIn score:

            For each checkpoint: $y = Wx,$ and $loss = (y - label)^2.$ Recall for this
            test case, there is no activation on y. For this example, $label = x.$

            Fast Rand Proj gives $\nabla_W loss = \nabla_y loss (x^T).$ We have $x$ and
            y as scalars so we can simply multiply. So then,

            \[\nabla_y loss * x = 2(y-x)*x = 2(Wx -x)*x = 2x^2 (w - 1).\]

            And we simply multiply these for x, x'. In this case, $x, x' \in [1..16]$.
            """
            for i in range(train_scores.shape[0]):
                for j in range(len(train_scores[0])):
                    _weights = torch.Tensor(weights)
                    num = 2 * (i + 1) * (i + 1) * (_weights - 1)
                    num *= 2 * (j + 1) * (j + 1) * (_weights - 1)
                    assertTensorAlmostEqual(
                        self, torch.sum(num), train_scores[i][j], delta=0.1
                    )

    def _test_tracin_identity_regression_setup(self, tmpdir: str):
        num_features = 7
        dataset = IdentityDataset(num_features)
        net = CoefficientNet()

        num_checkpoints = 5

        for i in range(num_checkpoints):
            net.fc1.weight.data = torch.rand((1, num_features))
            checkpoint_name = "-".join(["checkpoint-reg", str(i) + ".pt"])
            torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        return dataset, net

    @parameterized.expand(
        [
            ("check_idx", "none", DataInfluenceConstructor(TracInCP)),
            ("sample_wise_trick", None, DataInfluenceConstructor(TracInCP)),
            ("check_idx", "sum", DataInfluenceConstructor(TracInCPFast)),
            ("check_idx", "sum", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("check_idx", "mean", DataInfluenceConstructor(TracInCPFast)),
            ("check_idx", "mean", DataInfluenceConstructor(TracInCPFastRandProj)),
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_identity_regression(
        self, mode: str, reduction: Optional[str], tracin_constructor: Callable
    ) -> None:
        """
        This test uses a linear model with positive coefficients, where input feature
        matrix is the identity matrix.  Since the dot product between 2 different
        training instances is always 0, when calculating influence scores on the
        training data, only self influence scores will be nonzero.  Since the linear
        model has positive coefficients, self influence scores will be positive.
        Thus, the training instance with the largest influence on another training
        instance is itself.
        """

        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 4

            dataset, net = self._test_tracin_identity_regression_setup(tmpdir)

            train_inputs = dataset.samples
            train_labels = dataset.labels

            self.assertTrue(callable(tracin_constructor))

            if mode == "check_idx":

                self.assertTrue(isinstance(reduction, str))
                criterion = nn.MSELoss(reduction=cast(str, reduction))

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )

                # check influence scores of training data

                train_scores = tracin.influence(train_inputs, train_labels)
                idx, _ = tracin.influence(
                    train_inputs, train_labels, k=len(dataset), proponents=True
                )

                # check that top influence for an instance is itself
                for i in range(len(idx)):
                    self.assertEqual(idx[i][0], i)

            if mode == "sample_wise_trick":

                criterion = nn.MSELoss(reduction="none")

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=True,
                )

                train_scores = tracin.influence(train_inputs, train_labels)
                train_scores_tracin_sample_wise_trick = (
                    tracin_sample_wise_trick.influence(train_inputs, train_labels)
                )
                assertTensorAlmostEqual(
                    self, train_scores, train_scores_tracin_sample_wise_trick
                )
