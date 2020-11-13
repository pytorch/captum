from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from captum.optim._utils.typing import ModuleOutputMapping


class Loss(ABC):
    r"""
    Abstract Class to describe loss.
    """

    @abstractmethod
    def __init__(self, target: nn.Module):
        super(Loss, self).__init__()
        self.target = target

    @abstractmethod
    def __call__(self, x):
        pass


class ChannelActivation(Loss):
    """
    Maximize activations at the target layer and target channel.
    """

    def __init__(self, target: nn.Module, channel_index: int):
        super(Loss, self).__init__()
        self.target = target
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        assert activations is not None
        # ensure channel_index is valid
        assert self.channel_index < activations.shape[1]
        # assume NCHW
        # NOTE: not necessarily true e.g. for Linear layers
        # assert len(activations.shape) == 4
        return activations[:, self.channel_index, ...]


class NeuronActivation(Loss):
    def __init__(
        self, target: nn.Module, channel_index: int, x: int = None, y: int = None
    ):
        super(Loss, self).__init__()
        self.target = target
        self.channel_index = channel_index
        self.x = x
        self.y = y

        # ensure channel_index will be valid
        assert self.channel_index < self.target.out_channels

    def _call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        assert activations is not None
        assert len(activations.shape) == 4  # assume NCHW
        _, _, H, W = activations.shape

        if self.x is None:
            _x = W // 2
        else:
            assert self.x < W
            _x = self.x

        if self.y is None:
            _y = H // 2
        else:
            assert self.y < W
            _y = self.y

        return activations[:, self.channel_index, _x, _y]


class DeepDream(Loss):
    """
    Maximize 'interestingness' at the target layer.
    Mordvintsev et al., 2015.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        return activations ** 2


class TotalVariation(Loss):
    """
    Total variation denoising penalty for activations.
    See Simonyan, et al., 2014.
    """

    def _call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        x_diff = activations[..., 1:, :] - activations[..., :-1, :]
        y_diff = activations[..., :, 1:] - activations[..., :, :-1]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


class l1(Loss):
    """
    L1 norm of the target layer, generally used as a penalty.
    """

    def __init__(self, target: nn.Module, constant: float = 0):
        super(Loss, self).__init__()
        self.target = target
        self.constant = constant

    def _call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        return torch.abs(activations - self.constant).sum()


class l2(Loss):
    """
    L2 norm of the target layer, generally used as a penalty.
    """

    def __init__(self, target: nn.Module, constant: float = 0, epsilon: float = 1e-6):
        self.target = target
        self.constant = constant
        self.epsilon = epsilon

    def _call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        activations = (activations - self.constant).sum()
        return torch.sqrt(self.epsilon + activations)


class diversity(Loss):
    """
    Use a cosine similarity penalty to extract features from a polysemantic neuron.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#diversity
    """

    def _call__(self, targets_to_values: ModuleOutputMapping):
        activations = targets_to_values[self.target]
        return -sum(
            [
                sum(
                    [
                        (
                            torch.cosine_similarity(
                                activations[j].view(1, -1), activations[i].view(1, -1)
                            )
                        ).sum()
                        for i in range(activations.size(0))
                        if i != j
                    ]
                )
                for j in range(activations.size(0))
            ]
        ) / activations.size(0)
