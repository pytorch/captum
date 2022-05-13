import inspect
import os
import unittest
from functools import partial
from typing import Callable, Iterator, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.influence import DataInfluence
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from parameterized import parameterized
from parameterized.parameterized import param
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


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


class DataInfluenceConstructor:
    name: str = ""
    data_influence_class: type

    def __init__(
        self, data_influence_class: type, name: Optional[str] = None, **kwargs
    ):
        self.data_influence_class = data_influence_class
        self.name = name if name else data_influence_class.__name__
        self.kwargs = kwargs

    def __repr__(self):
        return self.name

    def __call__(
        self,
        net: Module,
        dataset: Union[Dataset, DataLoader],
        tmpdir: Union[str, List[str], Iterator],
        batch_size: Union[int, None],
        loss_fn: Optional[Union[Module, Callable]],
        **kwargs,
    ) -> DataInfluence:
        constuctor_kwargs = self.kwargs.copy()
        constuctor_kwargs.update(kwargs)
        if self.data_influence_class is TracInCPFastRandProj:
            self.check_annoy()
        if self.data_influence_class in [TracInCPFast, TracInCPFastRandProj]:
            return self.data_influence_class(
                net,
                list(net.children())[-1],
                dataset,
                tmpdir,
                loss_fn=loss_fn,
                batch_size=batch_size,
                **constuctor_kwargs,
            )
        else:
            return self.data_influence_class(
                net,
                dataset,
                tmpdir,
                batch_size=batch_size,
                loss_fn=loss_fn,
                **constuctor_kwargs,
            )

    def check_annoy(self) -> None:
        try:
            import annoy  # noqa
        except ImportError:
            raise unittest.SkipTest(
                (
                    f"Skipping tests for {self.data_influence_class.__name__}, "
                    "because it requires the Annoy module."
                )
            )


def generate_test_name(
    testcase_func: Callable,
    param_num: str,
    param: param,
    args_to_skip: Optional[List[str]] = None,
) -> str:
    """
    Creates human readable names for parameterized tests
    """

    if args_to_skip is None:
        args_to_skip = []
    param_strs = []

    func_param_names = list(inspect.signature(testcase_func).parameters)
    # skip the first 'self' parameter
    if func_param_names[0] == "self":
        func_param_names = func_param_names[1:]

    for i, arg in enumerate(param.args):
        if func_param_names[i] in args_to_skip:
            continue
        if isinstance(arg, bool):
            if arg:
                param_strs.append(func_param_names[i])
        else:
            args_str = str(arg)
            if args_str.isnumeric():
                param_strs.append(func_param_names[i])
            param_strs.append(args_str)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(param_strs)),
    )


def build_test_name_func(args_to_skip: Optional[List[str]] = None):
    """
    Returns function to generate human readable names for parameterized tests
    """

    return partial(generate_test_name, args_to_skip=args_to_skip)
