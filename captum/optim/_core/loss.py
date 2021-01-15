from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from captum.optim._utils.image.common import get_neuron_pos
from captum.optim._utils.typing import ModuleOutputMapping


class Loss(ABC):
    """
    Abstract Class to describe loss.
    """

    def __init__(self, target: nn.Module) -> None:
        super(Loss, self).__init__()
        self.target = target

    @abstractmethod
    def __call__(self, targets_to_values: ModuleOutputMapping):
        pass


class LayerActivation(Loss):
    """
    Maximize activations at the target layer.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        return targets_to_values[self.target]


class ChannelActivation(Loss):
    """
    Maximize activations at the target layer and target channel.
    """

    def __init__(self, target: nn.Module, channel_index: int) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
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
        self,
        target: nn.Module,
        channel_index: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.channel_index = channel_index
        self.x = x
        self.y = y

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        assert self.channel_index < activations.shape[1]
        assert len(activations.shape) == 4  # assume NCHW
        _x, _y = get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )

        return activations[:, self.channel_index, _x : _x + 1, _y : _y + 1]


class DeepDream(Loss):
    """
    Maximize 'interestingness' at the target layer.
    Mordvintsev et al., 2015.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        return activations ** 2


class TotalVariation(Loss):
    """
    Total variation denoising penalty for activations.
    See Mahendran, V. 2014. Understanding Deep Image Representations by Inverting Them.
    https://arxiv.org/abs/1412.0035
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        x_diff = activations[..., 1:, :] - activations[..., :-1, :]
        y_diff = activations[..., :, 1:] - activations[..., :, :-1]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


class L1(Loss):
    """
    L1 norm of the target layer, generally used as a penalty.
    """

    def __init__(self, target: nn.Module, constant: float = 0.0) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.constant = constant

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        return torch.abs(activations - self.constant).sum()


class L2(Loss):
    """
    L2 norm of the target layer, generally used as a penalty.
    """

    def __init__(
        self, target: nn.Module, constant: float = 0.0, epsilon: float = 1e-6
    ) -> None:
        self.target = target
        self.constant = constant
        self.epsilon = epsilon

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = (activations - self.constant).sum()
        return torch.sqrt(self.epsilon + activations)


class Diversity(Loss):
    """
    Use a cosine similarity penalty to extract features from a polysemantic neuron.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#diversity
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
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


class ActivationInterpolation(Loss):
    """
    Interpolate between two different layers & channels.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons
    """

    def __init__(
        self,
        target1: nn.Module = None,
        channel_index1: int = -1,
        target2: nn.Module = None,
        channel_index2: int = -1,
    ) -> None:
        super(Loss, self).__init__()
        self.target_one = target1
        self.channel_index_one = channel_index1
        self.target_two = target2
        self.channel_index_two = channel_index2

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations_one = targets_to_values[self.target_one]
        activations_two = targets_to_values[self.target_two]

        assert activations_one is not None and activations_two is not None
        # ensure channel indices are valid
        assert (
            self.channel_index_one < activations_one.shape[1]
            and self.channel_index_two < activations_two.shape[1]
        )
        assert activations_one.size(0) == activations_two.size(0)

        if self.channel_index_one > -1:
            activations_one = activations_one[:, self.channel_index_one]
        if self.channel_index_two > -1:
            activations_two = activations_two[:, self.channel_index_two]
        B = activations_one.size(0)

        batch_weights = torch.arange(B, device=activations_one.device) / (B - 1)
        sum_tensor = torch.zeros(1, device=activations_one.device)
        for n in range(B):
            sum_tensor = (
                sum_tensor + ((1 - batch_weights[n]) * activations_one[n]).mean()
            )
            sum_tensor = sum_tensor + (batch_weights[n] * activations_two[n]).mean()
        return sum_tensor


class Alignment(Loss):
    """
    Penalize the L2 distance between tensors in the batch to encourage visual
    similarity between them.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons
    """

    def __init__(self, target: nn.Module, decay_ratio: float = 2.0) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.decay_ratio = decay_ratio

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        B = activations.size(0)

        sum_tensor = torch.zeros(1, device=activations.device)
        for d in [1, 2, 3, 4]:
            for i in range(B - d):
                a, b = i, i + d
                activ_a, activ_b = activations[a], activations[b]
                sum_tensor = sum_tensor + (
                    (activ_a - activ_b) ** 2
                ).mean() / self.decay_ratio ** float(d)

        return sum_tensor


class Direction(Loss):
    """
    Visualize a general direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    """

    def __init__(self, target: nn.Module, vec: torch.Tensor) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec.reshape((1, -1, 1, 1))

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations.size(1) == self.direction.size(1)
        return torch.cosine_similarity(self.direction, activations)


class NeuronDirection(Loss):
    """
    Visualize a single (x, y) position for a direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        x: Optional[int] = None,
        y: Optional[int] = None,
        channel_index: Optional[int] = None,
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec.reshape((1, -1, 1, 1))
        self.x = x
        self.y = y
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        _x, _y = get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )
        activations = activations[:, :, _x : _x + 1, _y : _y + 1]
        if self.channel_index is not None:
            activations = activations[:, self.channel_index, ...][:, None, ...]
        return torch.cosine_similarity(self.direction, activations)


class WhitenedNeuronDirection(Loss):
    """
    Visualize a single (x, y) position for a direction vector &
    whitened activation samples.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/
    """

    def __init__(
        self,
        target: torch.nn.Module,
        vec: torch.Tensor,
        vec_w: torch.Tensor,
        cossim_pow: float = 4.0,
        x: Optional[int] = None,
        y: Optional[int] = None,
        eps: float = 1.0e-4,
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec.reshape((1, -1))
        self.vec_w = vec_w
        self.cossim_pow = cossim_pow
        self.eps = eps
        self.x = x
        self.y = y

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4
        assert activations.shape[0] == 1
        _y = activations.shape[2] // 2 if self.y is None else self.y
        _x = activations.shape[3] // 2 if self.x is None else self.x

        activations = activations[0, :, _x, _y]
        vec = torch.matmul(self.direction, self.vec_w)[0]
        if self.cossim_pow == 0:
            return activations * vec

        dot = torch.mean(activations * vec)
        cossims = dot / (self.eps + torch.sqrt(torch.sum(activations ** 2)))
        floored_cossims = torch.max(torch.full_like(cossims, 0.1), cossims)
        return dot * floored_cossims ** self.cossim_pow


class TensorDirection(Loss):
    """
    Visualize a tensor direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    """

    def __init__(self, target: nn.Module, vec: torch.Tensor) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        H_direction, W_direction = self.direction.size(2), self.direction.size(3)
        H_activ, W_activ = activations.size(2), activations.size(3)

        H = (H_activ - H_direction) // 2
        W = (W_activ - W_direction) // 2

        activations = activations[:, :, H : H + H_direction, W : W + W_direction]
        return torch.cosine_similarity(self.direction, activations)


class ActivationWeights(Loss):
    """
    Apply weights to channels, neurons, or spots in the target.
    """

    def __init__(
        self,
        target: nn.Module,
        weights: torch.Tensor = None,
        neuron: bool = False,
        x: Optional[int] = None,
        y: Optional[int] = None,
        wx: Optional[int] = None,
        wy: Optional[int] = None,
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.x = x
        self.y = y
        self.wx = wx
        self.wy = wy
        self.weights = weights
        self.neuron = x is not None or y is not None or neuron
        assert (
            wx is None
            and wy is None
            or wx is not None
            and wy is not None
            and x is not None
            and y is not None
        )

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        if self.neuron:
            assert activations.dim() == 4
            if self.wx is None and self.wy is None:
                _x, _y = get_neuron_pos(
                    activations.size(2), activations.size(3), self.x, self.y
                )
                activations = (
                    activations[..., _x : _x + 1, _y : _y + 1].squeeze() * self.weights
                )
            else:
                activations = activations[
                    ..., self.y : self.y + self.wy, self.x : self.x + self.wx
                ] * self.weights.view(1, -1, 1, 1)
        else:
            activations = activations * self.weights.view(1, -1, 1, 1)
        return activations
