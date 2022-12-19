import inspect
import os
import unittest
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple, Union

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
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


def _isSorted(x, key=lambda x: x, descending=True):
    if descending:
        return all([key(x[i]) >= key(x[i + 1]) for i in range(len(x) - 1)])
    else:
        return all([key(x[i]) <= key(x[i + 1]) for i in range(len(x) - 1)])


def _wrap_model_in_dataparallel(net):
    alt_device_ids = [0] + [x for x in range(torch.cuda.device_count() - 1, 0, -1)]
    net = net.cuda()
    return torch.nn.DataParallel(net, device_ids=alt_device_ids)


def _move_sample_to_cuda(samples):
    return [s.cuda() for s in samples]


class ExplicitDataset(Dataset):
    def __init__(self, samples, labels, use_gpu=False) -> None:
        self.samples, self.labels = samples, labels
        if use_gpu:
            self.samples = (
                _move_sample_to_cuda(self.samples)
                if isinstance(self.samples, list)
                else self.samples.cuda()
            )
            self.labels = self.labels.cuda()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


class UnpackDataset(Dataset):
    def __init__(self, samples, labels, use_gpu=False) -> None:
        self.samples, self.labels = samples, labels
        if use_gpu:
            self.samples = (
                _move_sample_to_cuda(self.samples)
                if isinstance(self.samples, list)
                else self.samples.cuda()
            )
            self.labels = self.labels.cuda()

    def __len__(self) -> int:
        return len(self.samples[0])

    def __getitem__(self, idx):
        """
        The signature of the returning item is: List[List], where the contents
        are: [sample_0, sample_1, ...] + [labels] (two lists concacenated).
        """
        return [lst[idx] for lst in self.samples] + [self.labels[idx]]


class IdentityDataset(ExplicitDataset):
    def __init__(self, num_features, use_gpu=False) -> None:
        self.samples = torch.diag(torch.ones(num_features))
        self.labels = torch.zeros(num_features).unsqueeze(1)
        if use_gpu:
            self.samples = self.samples.cuda()
            self.labels = self.labels.cuda()


class RangeDataset(ExplicitDataset):
    def __init__(self, low, high, num_features, use_gpu=False) -> None:
        self.samples = (
            torch.arange(start=low, end=high, dtype=torch.float)
            .repeat(num_features, 1)
            .transpose(1, 0)
        )
        self.labels = torch.arange(start=low, end=high, dtype=torch.float).unsqueeze(1)
        if use_gpu:
            self.samples = self.samples.cuda()
            self.labels = self.labels.cuda()


class BinaryDataset(ExplicitDataset):
    def __init__(self, use_gpu=False) -> None:
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
        super().__init__(self.samples, self.labels, use_gpu)


class CoefficientNet(nn.Module):
    def __init__(self, in_features=1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1, bias=False)
        self.fc1.weight.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        return x


class BasicLinearNet(nn.Module):
    def __init__(self, in_features, hidden_nodes, out_features) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, input):
        x = torch.tanh(self.linear1(input))
        return torch.tanh(self.linear2(x))


class MultLinearNet(nn.Module):
    def __init__(self, in_features, hidden_nodes, out_features, num_inputs) -> None:
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


def get_random_model_and_data(
    tmpdir, unpack_inputs, return_test_data=True, use_gpu=False
):

    in_features, hidden_nodes, out_features = 5, 4, 3
    num_inputs = 2

    net = (
        BasicLinearNet(in_features, hidden_nodes, out_features)
        if not unpack_inputs
        else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
    ).double()

    num_checkpoints = 5

    for i in range(num_checkpoints):
        net.linear1.weight.data = torch.normal(
            3, 4, (hidden_nodes, in_features)
        ).double()
        net.linear2.weight.data = torch.normal(
            5, 6, (out_features, hidden_nodes)
        ).double()
        if unpack_inputs:
            net.pre.weight.data = torch.normal(
                3, 4, (in_features, in_features * num_inputs)
            )
        if hasattr(net, "pre"):
            net.pre.weight.data = net.pre.weight.data.double()
        checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

    num_samples = 50
    num_train = 32
    all_labels = torch.normal(1, 2, (num_samples, out_features)).double()
    train_labels = all_labels[:num_train]
    test_labels = all_labels[num_train:]

    if unpack_inputs:
        all_samples = [
            torch.normal(0, 1, (num_samples, in_features)).double()
            for _ in range(num_inputs)
        ]
        train_samples = [ts[:num_train] for ts in all_samples]
        test_samples = [ts[num_train:] for ts in all_samples]
    else:
        all_samples = torch.normal(0, 1, (num_samples, in_features)).double()
        train_samples = all_samples[:num_train]
        test_samples = all_samples[num_train:]

    dataset = (
        ExplicitDataset(train_samples, train_labels, use_gpu)
        if not unpack_inputs
        else UnpackDataset(train_samples, train_labels, use_gpu)
    )

    if return_test_data:
        return (
            _wrap_model_in_dataparallel(net) if use_gpu else net,
            dataset,
            _move_sample_to_cuda(test_samples)
            if isinstance(test_samples, list) and use_gpu
            else test_samples.cuda()
            if use_gpu
            else test_samples,
            test_labels.cuda() if use_gpu else test_labels,
        )
    else:
        return _wrap_model_in_dataparallel(net) if use_gpu else net, dataset


class DataInfluenceConstructor:
    name: str = ""
    data_influence_class: type

    def __init__(
        self,
        data_influence_class: type,
        name: Optional[str] = None,
        duplicate_loss_fn: bool = False,
        **kwargs,
    ) -> None:
        """
        if `duplicate_loss_fn` is True, will explicitly pass the provided `loss_fn` as
        the `test_loss_fn` when constructing the TracInCPBase instance
        """
        self.data_influence_class = data_influence_class
        self.name = name if name else data_influence_class.__name__
        self.duplicate_loss_fn = duplicate_loss_fn
        self.kwargs = kwargs

    def __repr__(self) -> str:
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
        constructor_kwargs = self.kwargs.copy()
        constructor_kwargs.update(kwargs)
        # if `self.duplicate_loss_fn`, explicitly pass in `loss_fn` as `test_loss_fn`
        # when constructing the instance. Doing so should not affect the behavior of
        # the returned tracincp instance, since if `test_loss_fn` is not passed in,
        # the constructor sets `test_loss_fn` to be the same as `loss_fn`
        if self.duplicate_loss_fn:
            constructor_kwargs["test_loss_fn"] = loss_fn
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
                **constructor_kwargs,
            )
        else:
            return self.data_influence_class(
                net,
                dataset,
                tmpdir,
                batch_size=batch_size,
                loss_fn=loss_fn,
                **constructor_kwargs,
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


def _format_batch_into_tuple(
    inputs: Union[Tuple, Tensor], targets: Tensor, unpack_inputs: bool
):
    if unpack_inputs:
        return (*inputs, targets)
    else:
        return (inputs, targets)
