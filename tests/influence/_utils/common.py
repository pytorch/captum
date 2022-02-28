import os
import tempfile
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.helpers.basic import BaseTest
from tests.helpers.basic import assertTensorAlmostEqual
from torch.utils.data import Dataset, DataLoader


def isSorted(x, key=lambda x: x, descending=True):
    if descending:
        return all([key(x[i]) >= key(x[i + 1]) for i in range(len(x) - 1)])
    else:
        return all([key(x[i]) <= key(x[i + 1]) for i in range(len(x) - 1)])


class ExplicitDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples, self.labels = samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


class UnpackDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples, self.labels = samples, labels

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, idx):
        """
        The signature of the returning item is: List[List], where the contents
        are: [sample_0, sample_1, ...] + [labels] (two lists concacenated).
        """
        return [lst[idx] for lst in self.samples] + [self.labels[idx]]


class IdentityDataset(ExplicitDataset):
    def __init__(self, num_features):
        self.samples = torch.diag(torch.ones(num_features))
        self.labels = torch.zeros(num_features).unsqueeze(1)


class RangeDataset(ExplicitDataset):
    def __init__(self, low, high, num_features):
        self.samples = (
            torch.arange(start=low, end=high, dtype=torch.float)
            .repeat(num_features, 1)
            .transpose(1, 0)
        )
        self.labels = torch.arange(start=low, end=high, dtype=torch.float).unsqueeze(1)


class BinaryDataset(ExplicitDataset):
    def __init__(self):
        self.samples = F.normalize(
            torch.stack(
                (
                    torch.Tensor([1, 1]),
                    torch.Tensor([2, 1]),
                    torch.Tensor([1, 2]),
                    torch.Tensor([1, 5]),
                    torch.Tensor([0.01, 1]),
                    torch.Tensor([5, 1]),
                    torch.Tensor([1, 0.01]),
                    torch.Tensor([-1, -1]),
                    torch.Tensor([-2, -1]),
                    torch.Tensor([-1, -2]),
                    torch.Tensor([-1, -5]),
                    torch.Tensor([-5, -1]),
                    torch.Tensor([1, -1]),
                    torch.Tensor([2, -1]),
                    torch.Tensor([1, -2]),
                    torch.Tensor([1, -5]),
                    torch.Tensor([0.01, -1]),
                    torch.Tensor([5, -1]),
                    torch.Tensor([-1, 1]),
                    torch.Tensor([-2, 1]),
                    torch.Tensor([-1, 2]),
                    torch.Tensor([-1, 5]),
                    torch.Tensor([-5, 1]),
                    torch.Tensor([-1, 0.01]),
                )
            )
        )
        self.labels = torch.cat(
            (
                torch.Tensor([1]).repeat(12, 1),
                torch.Tensor([-1]).repeat(12, 1),
            )
        )


class CoefficientNet(nn.Module):
    def __init__(self, in_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1, bias=False)
        self.fc1.weight.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        return x


class BasicLinearNet(nn.Module):
    def __init__(self, in_features, hidden_nodes, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, input):
        x = torch.tanh(self.linear1(input))
        return torch.tanh(self.linear2(x))


class MultLinearNet(nn.Module):
    def __init__(self, in_features, hidden_nodes, out_features, num_inputs):
        super().__init__()
        self.pre = nn.Linear(in_features * num_inputs, in_features)
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, *inputs):
        """
        The signature of inputs is List[torch.Tensor],
        where torch.Tensor has the dimensions [num_inputs x in_features].
        It first concacenates the list and a linear layer to reduce the
        dimension.
        """
        inputs = self.pre(torch.cat(inputs, dim=1))
        x = torch.tanh(self.linear1(inputs))
        return torch.tanh(self.linear2(x))


class _TestTracInRegression:
    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

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

    def _test_tracin_regression(self, features: int, mode: int) -> None:
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

            assert callable(self.tracin_constructor)

            if mode == "check_idx":

                assert isinstance(self.reduction, str)
                criterion = nn.MSELoss(reduction=self.reduction)

                tracin = self.tracin_constructor(
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
                assert isinstance(self, BaseTest)
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

                tracin = self.tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = self.tracin_constructor(
                    net, dataset, tmpdir, batch_size, criterion, True
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


class _TestTracInRegression1DCheckIdx(_TestTracInRegression):
    def test_tracin_regression_1D_check_idx(self):
        self._test_tracin_regression(1, "check_idx")


class _TestTracInRegression1DCheckSampleWiseTrick(_TestTracInRegression):
    def test_tracin_regression_1D_check_sample_wise_trick(self):
        self._test_tracin_regression(1, "sample_wise_trick")


class _TestTracInRegression20DCheckIdx(_TestTracInRegression):
    def test_tracin_regression_20D_check_idx(self):
        self._test_tracin_regression(20, "check_idx")


class _TestTracInRegression20DCheckSampleWiseTrick(_TestTracInRegression):
    def test_tracin_regression_20D_check_sample_wise_trick(self):
        self._test_tracin_regression(20, "sample_wise_tricksample_wise_trick")


class _TestTracInXOR:
    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

    def _test_tracin_xor_setup(self, tmpdir: str):
        net = BasicLinearNet(2, 2, 1)

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.2956, -1.4465], [-0.3890, -0.7420]]),
                ),
                ("linear1.bias", torch.Tensor([1.2924, 0.0021])),
                ("linear2.weight", torch.Tensor([[-1.2013, 0.7174]])),
                ("linear2.bias", torch.Tensor([0.5880])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "0" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.3238, -1.4899], [-0.4544, -0.7448]]),
                ),
                ("linear1.bias", torch.Tensor([1.3185, -0.0317])),
                ("linear2.weight", torch.Tensor([[-1.2342, 0.7741]])),
                ("linear2.bias", torch.Tensor([0.6234])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "1" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.3546, -1.5288], [-0.5250, -0.7591]]),
                ),
                ("linear1.bias", torch.Tensor([1.3432, -0.0684])),
                ("linear2.weight", torch.Tensor([[-1.2490, 0.8534]])),
                ("linear2.bias", torch.Tensor([0.6749])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "2" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.4022, -1.5485], [-0.5688, -0.7607]]),
                ),
                ("linear1.bias", torch.Tensor([1.3740, -0.1571])),
                ("linear2.weight", torch.Tensor([[-1.3412, 0.9013]])),
                ("linear2.bias", torch.Tensor([0.6468])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "3" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.4464, -1.5890], [-0.6348, -0.7665]]),
                ),
                ("linear1.bias", torch.Tensor([1.3791, -0.2008])),
                ("linear2.weight", torch.Tensor([[-1.3818, 0.9586]])),
                ("linear2.bias", torch.Tensor([0.6954])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "4" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.5217, -1.6242], [-0.6644, -0.7842]]),
                ),
                ("linear1.bias", torch.Tensor([1.3500, -0.2418])),
                ("linear2.weight", torch.Tensor([[-1.4304, 0.9980]])),
                ("linear2.bias", torch.Tensor([0.7567])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "5" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.5551, -1.6631], [-0.7420, -0.8025]]),
                ),
                ("linear1.bias", torch.Tensor([1.3508, -0.2618])),
                ("linear2.weight", torch.Tensor([[-1.4272, 1.0772]])),
                ("linear2.bias", torch.Tensor([0.8427])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "6" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.5893, -1.6656], [-0.7863, -0.8369]]),
                ),
                ("linear1.bias", torch.Tensor([1.3949, -0.3215])),
                ("linear2.weight", torch.Tensor([[-1.4555, 1.1600]])),
                ("linear2.bias", torch.Tensor([0.8730])),
            ]
        )
        net.load_state_dict(state)
        checkpoint_name = "-".join(["checkpoint", "class", "7" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        dataset = BinaryDataset()

        return net, dataset

    def _test_tracin_xor(self, mode) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = BinaryDataset()
            net = BasicLinearNet(2, 2, 1)

            batch_size = 4

            net, dataset = self._test_tracin_xor_setup(tmpdir)

            testset = F.normalize(torch.empty(100, 2).normal_(mean=0, std=0.5), dim=1)
            mask = ~torch.logical_xor(testset[:, 0] > 0, testset[:, 1] > 0)
            testlabels = (
                torch.where(mask, torch.tensor(1), torch.tensor(-1))
                .unsqueeze(1)
                .float()
            )

            assert isinstance(self, BaseTest)
            assert callable(self.tracin_constructor)

            if mode == "check_idx":

                assert isinstance(self.reduction, str)
                criterion = nn.MSELoss(reduction=self.reduction)

                tracin = self.tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )
                test_scores = tracin.influence(testset, testlabels)
                idx = torch.argsort(test_scores, dim=1, descending=True)

                # check that top 5 influences have matching binary classification
                for i in range(len(idx)):
                    influence_labels = dataset.labels[idx[i][0:5], 0]
                    self.assertTrue(torch.all(testlabels[i, 0] == influence_labels))

            if mode == "sample_wise_trick":

                criterion = nn.MSELoss(reduction="none")

                tracin = self.tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = self.tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    True,
                )

                test_scores = tracin.influence(testset, testlabels)
                test_scores_sample_wise_trick = tracin_sample_wise_trick.influence(
                    testset, testlabels
                )
                assertTensorAlmostEqual(
                    self, test_scores, test_scores_sample_wise_trick
                )


class _TestTracInXORCheckIdx(_TestTracInXOR):
    def test_tracin_xor_check_idx(self):
        self._test_tracin_xor("check_idx")


class _TestTracInXORCheckSampleWiseTrick(_TestTracInXOR):
    def test_tracin_xor_check_sample_wise_trick(self):
        self._test_tracin_xor("sample_wise_trick")


class _TestTracInIdentityRegression:
    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

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

    def _test_tracin_identity_regression(self, mode) -> None:
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

            assert callable(self.tracin_constructor)
            assert isinstance(self, BaseTest)

            if mode == "check_idx":

                assert isinstance(self.reduction, str)
                criterion = nn.MSELoss(reduction=self.reduction)

                tracin = self.tracin_constructor(
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

                tracin = self.tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = self.tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    True,
                )

                train_scores = tracin.influence(train_inputs, train_labels)
                train_scores_tracin_sample_wise_trick = (
                    tracin_sample_wise_trick.influence(train_inputs, train_labels)
                )
                assertTensorAlmostEqual(
                    self, train_scores, train_scores_tracin_sample_wise_trick
                )


class _TestTracInIdentityRegressionCheckIdx(_TestTracInIdentityRegression):
    def test_tracin_identity_regression_check_idx(self):
        self._test_tracin_identity_regression("check_idx")


class _TestTracInIdentityRegressionCheckSampleWiseTrick(_TestTracInIdentityRegression):
    def test_tracin_identity_regression_check_sample_wise_trick(self):
        self._test_tracin_identity_regression("sample_wise_trick")


class _TestTracInRandomProjectionRegression:
    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

    def _test_tracin_random_projection_regression_setup(self, tmpdir: str):

        low = 1
        high = 17
        features = 20
        dataset = RangeDataset(low, high, features)
        net = CoefficientNet(in_features=features)
        weights = [0.4379, 0.1653, 0.5132, 0.3651, 0.9992]

        for i, weight in enumerate(weights):
            net.fc1.weight.data.fill_(weight)
            checkpoint_name = "-".join(["checkpoint-reg", str(i) + ".pt"])
            torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        return dataset, net

    def test_tracin_random_projection_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 4

            dataset, net = self._test_tracin_random_projection_regression_setup(tmpdir)

            # check influence scores of training data
            assert isinstance(self.reduction, str)
            assert callable(self.tracin_constructor)
            assert isinstance(self, BaseTest)

            criterion = nn.MSELoss(reduction=self.reduction)
            tracin = self.tracin_constructor(
                net,
                dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            train_inputs = dataset.samples
            train_labels = dataset.labels
            idx, _ = tracin.influence(
                train_inputs, train_labels, k=len(dataset), proponents=True
            )
            # check that top influence is one with maximal value (and hence gradient)
            for i in range(len(idx)):
                self.assertEqual(idx[i][0], 15)

            # check influence scores of test data
            test_inputs = torch.arange(17, 33, dtype=torch.float).unsqueeze(1)
            test_labels = test_inputs
            test_inputs = test_inputs.repeat(1, train_inputs.shape[1])

            idx, _ = tracin.influence(
                test_inputs, test_labels, k=len(train_inputs), proponents=True
            )
            # check that top influence is one with maximal value (and hence gradient)
            for i in range(len(idx)):
                self.assertTrue(isSorted(idx[i]))


class _TestTracInRegression1DNumerical:
    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

    def test_tracin_regression_1D_numerical(self) -> None:

        low = 1
        high = 17
        features = 1
        dataset = RangeDataset(low, high, features)
        net = CoefficientNet()
        assert isinstance(self.reduction, str)
        criterion = nn.MSELoss(reduction=self.reduction)
        batch_size = 4
        weights = [0.4379, 0.1653, 0.5132, 0.3651, 0.9992]

        train_inputs = dataset.samples
        train_labels = dataset.labels

        with tempfile.TemporaryDirectory() as tmpdir:

            for i, weight in enumerate(weights):
                net.fc1.weight.data.fill_(weight)
                checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
                torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

            assert callable(self.tracin_constructor)
            tracin = self.tracin_constructor(
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


def get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True):

    in_features, hidden_nodes, out_features = 5, 4, 3
    num_inputs = 2

    net = (
        BasicLinearNet(in_features, hidden_nodes, out_features)
        if not unpack_inputs
        else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
    )

    num_checkpoints = 5

    for i in range(num_checkpoints):
        net.linear1.weight.data = torch.normal(3, 4, (hidden_nodes, in_features))
        net.linear2.weight.data = torch.normal(5, 6, (out_features, hidden_nodes))
        if unpack_inputs:
            net.pre.weight.data = torch.normal(
                3, 4, (in_features, in_features * num_inputs)
            )
        checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

    num_samples = 50
    num_train = 32
    all_labels = torch.normal(1, 2, (num_samples, out_features))
    train_labels = all_labels[:num_train]
    test_labels = all_labels[num_train:]

    if unpack_inputs:
        all_samples = [
            torch.normal(0, 1, (num_samples, in_features)) for _ in range(num_inputs)
        ]
        train_samples = [ts[:num_train] for ts in all_samples]
        test_samples = [ts[num_train:] for ts in all_samples]
    else:
        all_samples = torch.normal(0, 1, (num_samples, in_features))
        train_samples = all_samples[:num_train]
        test_samples = all_samples[num_train:]

    dataset = (
        ExplicitDataset(train_samples, train_labels)
        if not unpack_inputs
        else UnpackDataset(train_samples, train_labels)
    )

    if return_test_data:
        return net, dataset, test_samples, test_labels
    else:
        return net, dataset


class _TestTracInGetKMostInfluential:
    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

    def _test_tracin_get_k_most_influential(
        self, batch_size, k, unpack_inputs, proponents
    ):
        """
        This test constructs a random BasicLinearNet, and checks that the proponents
        obtained by calling `influence` and sorting are equal to the proponents
        obtained by calling `_get_k_most_influential`.  Those calls are made through
        the calls to wrapper method `influence`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True)

            assert isinstance(self.reduction, str)
            assert callable(self.tracin_constructor)

            criterion = nn.MSELoss(reduction=self.reduction)

            tracin = self.tracin_constructor(
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

    def _test_tracin_get_k_most_influential_batch_sizes(
        self, unpack_inputs=False, proponents=True
    ):
        """
        calls test helper method `_test_tracin_get_k_most_influential` for several
        combinations of `batch_size` and `k`.  This is important because the behavior
        of `_get_k_most_influential` depends on whether `k` is larger than
        `batch_size`.
        """
        for (batch_size, k) in [(4, 7), (7, 4), (40, 5), (5, 40), (40, 45)]:
            self._test_tracin_get_k_most_influential(
                batch_size, k, unpack_inputs, proponents
            )

    def test_tracin_get_k_most_influential_proponents(self):
        self._test_tracin_get_k_most_influential_batch_sizes(
            unpack_inputs=False, proponents=True
        )

    def test_tracin_get_k_most_influential_unpack_proponents(self):
        self._test_tracin_get_k_most_influential_batch_sizes(
            unpack_inputs=True, proponents=True
        )

    def test_tracin_get_k_most_influential_opponents(self):
        self._test_tracin_get_k_most_influential_batch_sizes(
            unpack_inputs=False, proponents=False
        )

    def test_tracin_get_k_most_influential_unpack_opponents(self):
        self._test_tracin_get_k_most_influential_batch_sizes(
            unpack_inputs=True, proponents=False
        )


class _TestTracInSelfInfluence:
    def _test_tracin_self_influence(self, unpack_inputs=False):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=False)

            # compute tracin_scores of training data on training data
            criterion = nn.MSELoss(reduction=self.reduction)
            batch_size = 5

            tracin = self.tracin_constructor(
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
            self_tracin_scores = tracin.influence()

            assertTensorAlmostEqual(
                self,
                torch.diagonal(train_scores),
                self_tracin_scores,
                delta=0.01,
                mode="max",
            )

    def test_tracin_self_influence(self):
        self._test_tracin_self_influence(unpack_inputs=False)

    def test_tracin_self_influence_unpack(self):
        self._test_tracin_self_influence(unpack_inputs=True)


class _TestTracInDataLoader:
    """
    This tests that the influence score computed when a Dataset is fed to the
    `self.tracin_constructor` and when a DataLoader constructed using the same
    Dataset is fed to `self.tracin_constructor` gives the same results.
    """

    reduction: Optional[str] = None
    tracin_constructor: Optional[Callable] = None

    def _test_tracin_dataloader_helper(self, unpack_inputs):

        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 5

            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True)

            assert isinstance(self.reduction, str)
            criterion = nn.MSELoss(reduction=self.reduction)

            assert callable(self.tracin_constructor)
            tracin = self.tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            train_scores = tracin.influence(
                test_samples, test_labels, k=None, unpack_inputs=unpack_inputs
            )

            tracin_dataloader = self.tracin_constructor(
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

    def test_tracin_dataloader(self):
        self._test_tracin_dataloader_helper(unpack_inputs=False)

    def test_tracin_dataloader_unpack(self):
        self._test_tracin_dataloader_helper(unpack_inputs=True)
