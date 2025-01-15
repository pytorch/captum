#!/usr/bin/env python3

# pyre-strict
import torch
import torch.nn as nn


class SigmoidModel(nn.Module):
    """
    Model architecture from:
    https://medium.com/coinmonks/create-a-neural-network-in
        -pytorch-and-make-your-life-simpler-ec5367895199
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, num_in, num_hidden, num_out) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_in = num_in
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_hidden = num_hidden
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_out)
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, input):
        lin1 = self.lin1(input)
        lin2 = self.lin2(self.relu1(lin1))
        return self.sigmoid(lin2)


class SoftmaxModel(nn.Module):
    """
    Model architecture from:
    https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, num_in, num_hidden, num_out, inplace: bool = False) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_in = num_in
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_hidden = num_hidden
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_out)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.softmax = nn.Softmax(dim=1)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, input):
        lin1 = self.relu1(self.lin1(input))
        lin2 = self.relu2(self.lin2(lin1))
        lin3 = self.lin3(lin2)
        return self.softmax(lin3)


class SigmoidDeepLiftModel(nn.Module):
    """
    Model architecture from:
    https://medium.com/coinmonks/create-a-neural-network-in
        -pytorch-and-make-your-life-simpler-ec5367895199
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, num_in, num_hidden, num_out) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_in = num_in
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_hidden = num_hidden
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden, bias=False)
        self.lin2 = nn.Linear(num_hidden, num_out, bias=False)
        self.lin1.weight = nn.Parameter(torch.ones(num_hidden, num_in))
        self.lin2.weight = nn.Parameter(torch.ones(num_out, num_hidden))
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, input):
        lin1 = self.lin1(input)
        lin2 = self.lin2(self.relu1(lin1))
        return self.sigmoid(lin2)


class SoftmaxDeepLiftModel(nn.Module):
    """
    Model architecture from:
    https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, num_in, num_hidden, num_out) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_in = num_in
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_hidden = num_hidden
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_out)
        self.lin1.weight = nn.Parameter(torch.ones(num_hidden, num_in))
        self.lin2.weight = nn.Parameter(torch.ones(num_hidden, num_hidden))
        self.lin3.weight = nn.Parameter(torch.ones(num_out, num_hidden))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, input):
        lin1 = self.relu1(self.lin1(input))
        lin2 = self.relu2(self.lin2(lin1))
        lin3 = self.lin3(lin2)
        return self.softmax(lin3)
