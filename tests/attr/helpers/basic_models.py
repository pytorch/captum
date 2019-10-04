#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input = 1 - F.relu(1 - input)
        return input


class BasicModel2(nn.Module):
    """
        Example model one from the paper
        https://arxiv.org/pdf/1703.01365.pdf

        f(x1, x2) = RELU(ReLU(x1) - 1 - ReLU(x2))
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        relu_out1 = F.relu(input1)
        relu_out2 = F.relu(input2)
        return F.relu(relu_out1 - 1 - relu_out2)


class BasicModel3(nn.Module):
    """
        Example model two from the paper
        https://arxiv.org/pdf/1703.01365.pdf

        f(x1, x2) = RELU(ReLU(x1 - 1) - ReLU(x2))
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        relu_out1 = F.relu(input1 - 1)
        relu_out2 = F.relu(input2)
        return F.relu(relu_out1 - relu_out2)


class BasicModel4_MultiArgs(nn.Module):
    """
        Slightly modified example model from the paper
        https://arxiv.org/pdf/1703.01365.pdf
        f(x1, x2) = RELU(ReLU(x1 - 1) - ReLU(x2) / x3)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2, additional_input1, additional_input2=0):
        relu_out1 = F.relu(input1 - 1)
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2.div(additional_input1)
        return F.relu(relu_out1 - relu_out2)[:, additional_input2]


class BasicModel5_MultiArgs(nn.Module):
    """
        Slightly modified example model from the paper
        https://arxiv.org/pdf/1703.01365.pdf
        f(x1, x2) = RELU(ReLU(x1 - 1) * x3[0] - ReLU(x2) * x3[1])
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2, additional_input1, additional_input2=0):
        relu_out1 = F.relu(input1 - 1) * additional_input1[0]
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2 * additional_input1[1]
        return F.relu(relu_out1 - relu_out2)[:, additional_input2]


class BasicModel6_MultiTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        input = input1 + input2
        return 1 - F.relu(1 - input)[:, 1]


class ReLUDeepLiftModel(nn.Module):
    r"""
        https://www.youtube.com/watch?v=f_iAM0NPwnM
    """

    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x1, x2):
        return 2 * self.relu1(x1) + 2 * self.relu2(x2 - 1.5)


class ReLULinearDeepLiftModel(nn.Module):
    r"""
        Architecture is based on:
        https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/
        tests/test_tensorflow.py#L65
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 1, bias=False)
        self.l2 = nn.Linear(3, 1, bias=False)
        self.l1.weight = nn.Parameter(torch.tensor([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]]))
        self.l2.weight = nn.Parameter(torch.tensor([[2.0, 3.0, 0.0], [0.0, 1.0, 2.0]]))
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        return self.relu(torch.cat([self.l1(x1), self.l2(x2)], axis=1)).sum(axis=1)


class TextModule(nn.Module):
    r"""Basic model that has inner embedding layer. This layer can be pluged
    into a larger network such as `BasicEmbeddingModel` and help us to test
    nested embedding layers
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.inner_embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input):
        return self.inner_embedding(input)


class BasicEmbeddingModel(nn.Module):
    r"""
    Implements basic model with nn.Embedding layer. This simple model
    will help us to test nested InterpretableEmbedding layers
    The model has the following structure:
    BasicEmbeddingModel(
      (embedding1): Embedding(30, 100)
      (embedding2): TextModule(
        (inner_embedding): Embedding(30, 100)
      )
      (linear1): Linear(in_features=100, out_features=256, bias=True)
      (relu): ReLU()
      (linear2): Linear(in_features=256, out_features=1, bias=True)
    )
    """

    def __init__(
        self, num_embeddings=30, embedding_dim=100, hidden_dim=256, output_dim=1
    ):
        super().__init__()
        self.embedding1 = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding2 = TextModule(num_embeddings, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        embedding1 = self.embedding1(input)
        embedding2 = self.embedding2(input)
        embeddings = embedding1 + embedding2
        return self.linear2(self.relu(self.linear1(embeddings))).squeeze(1)


class TestModel_MultiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear 0 is simply identity transform
        self.linear0 = nn.Linear(3, 3)
        self.linear0.weight = nn.Parameter(torch.eye(3))
        self.linear0.bias = nn.Parameter(torch.zeros(3))
        self.linear1 = nn.Linear(3, 4)
        self.linear1.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, 2)
        self.linear2.weight = nn.Parameter(torch.ones(2, 4))
        self.linear2.bias = nn.Parameter(torch.tensor([-1.0, 1.0]))

    def forward(self, x, add_input=None):
        input = x if add_input is None else x + add_input
        lin0_out = self.linear0(input)
        lin1_out = self.linear1(lin0_out)
        relu_out = self.relu(lin1_out)
        lin2_out = self.linear2(relu_out)
        return lin2_out


class TestModel_MultiLayer_MultiInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TestModel_MultiLayer()

    def forward(self, x1, x2, x3, scale):
        return self.model(scale * (x1 + x2 + x3))


class BasicModel_SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(8, 4)
        self.conv1.weight = nn.Parameter(torch.ones(2, 1, 3, 3))
        self.conv1.bias = nn.Parameter(torch.tensor([-50.0, -75.0]))
        self.fc1.weight = nn.Parameter(
            torch.cat([torch.ones(4, 5), -1 * torch.ones(4, 3)], dim=1)
        )
        self.relu2 = nn.ReLU()

    def forward(self, x, x2=None):
        if x2 is not None:
            x = x + x2
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 8)
        return self.relu2(self.fc1(x))


class TestModel_ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class TestModel_ConvNet_MaxPool1d(nn.Module):
    """Same as above, but with the MaxPool2d replaced
    with a MaxPool1d. This is useful because the MaxPool modules
    behave differently to other modules from the perspective
    of the DeepLift Attributions
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 2, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(2, 4, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class TestModel_ConvNet_MaxPool3d(nn.Module):
    """Same as above, but with the MaxPool1d replaced
    with a MaxPool3d. This is useful because the MaxPool modules
    behave differently to other modules from the perspective
    of the DeepLift Attributions
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(2, 4, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(4, 8)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight = nn.Parameter(torch.ones(8, 4))
        self.fc2.weight = nn.Parameter(torch.ones(10, 8))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
