import functools
import operator
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from captum.optim._utils.image.common import _dot_cossim, get_neuron_pos
from captum.optim._utils.typing import ModuleOutputMapping


def _make_arg_str(arg: Any) -> str:
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return arg[:15] + "..." if too_big else arg


class Loss(ABC):
    """
    Abstract Class to describe loss.
    Note: All Loss classes should expose self.target for hooking by
    InputOptimization
    """

    def __init__(self) -> None:
        super(Loss, self).__init__()

    @abstractproperty
    def target(self) -> Union[nn.Module, List[nn.Module]]:
        pass

    @abstractmethod
    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return self.__name__

    def __neg__(self) -> "CompositeLoss":
        return module_op(self, None, operator.neg)

    def __add__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.add)

    def __sub__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.sub)

    def __mul__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.mul)

    def __truediv__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.truediv)

    def __pow__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.pow)

    def __radd__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__add__(other)

    def __rsub__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__neg__().__add__(other)

    def __rmul__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        if isinstance(other, (int, float)):

            def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
                return operator.truediv(other, torch.mean(self(module)))

            name = self.__name__
            target = self.target
        elif isinstance(other, Loss):
            # This should never get called because __div__ will be called instead
            pass
        else:
            raise TypeError(
                "Can only apply math operations with int, float or Loss. Received type "
                + str(type(other))
            )
        return CompositeLoss(loss_fn, name=name, target=target)

    def __rpow__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        if isinstance(other, (int, float)):

            def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
                return operator.pow(other, torch.mean(self(module)))

            name = self.__name__
            target = self.target
        elif isinstance(other, Loss):
            # This should never get called because __pow__ will be called instead
            pass
        else:
            raise TypeError(
                "Can only apply math operations with int, float or Loss. Received type "
                + str(type(other))
            )
        return CompositeLoss(loss_fn, name=name, target=target)


def module_op(
    self: Loss, other: Union[None, int, float, Loss], math_op: Callable
) -> "CompositeLoss":
    """
    This is a general function for applying math operations to Losses
    """
    if other is None and math_op == operator.neg:

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(self(module))

        name = self.__name__
        target = self.target
    elif isinstance(other, (int, float)):

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(self(module), other)

        name = self.__name__
        target = self.target
    elif isinstance(other, Loss):
        # We take the mean of the output tensor to resolve shape mismatches
        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(torch.mean(self(module)), torch.mean(other(module)))

        name = f"Compose({', '.join([self.__name__, other.__name__])})"
        target = (self.target if isinstance(self.target, list) else [self.target]) + (
            other.target if isinstance(other.target, list) else [other.target]
        )

        # Filter out duplicate targets
        target = list(dict.fromkeys(target))
    else:
        raise TypeError(
            "Can only apply math operations with int, float or Loss. Received type "
            + str(type(other))
        )
    return CompositeLoss(loss_fn, name=name, target=target)


class BaseLoss(Loss):
    def __init__(
        self,
        target: Union[nn.Module, List[nn.Module]] = [],
        batch_index: Optional[int] = None,
    ) -> None:
        super(BaseLoss, self).__init__()
        self._target = target
        if batch_index is None:
            self._batch_index = (None, None)
        else:
            self._batch_index = (batch_index, batch_index + 1)

    @property
    def target(self) -> Union[nn.Module, List[nn.Module]]:
        return self._target

    @property
    def batch_index(self) -> Tuple:
        return self._batch_index


class CompositeLoss(BaseLoss):
    def __init__(
        self,
        loss_fn: Callable,
        name: str = "",
        target: Union[nn.Module, List[nn.Module]] = [],
    ) -> None:
        super(CompositeLoss, self).__init__(target)
        self.__name__ = name
        self.loss_fn = loss_fn

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        return self.loss_fn(targets_to_values)


def loss_wrapper(cls: Any) -> Callable:
    """
    Primarily for naming purposes.
    """

    @functools.wraps(cls)
    def wrapper(*args, **kwargs) -> object:
        obj = cls(*args, **kwargs)
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        obj.__name__ = cls.__name__ + args_str
        return obj

    return wrapper


@loss_wrapper
class LayerActivation(BaseLoss):
    """
    Maximize activations at the target layer.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return activations


@loss_wrapper
class ChannelActivation(BaseLoss):
    """
    Maximize activations at the target layer and target channel.
    """

    def __init__(
        self, target: nn.Module, channel_index: int, batch_index: Optional[int] = None
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        # ensure channel_index is valid
        assert self.channel_index < activations.shape[1]
        # assume NCHW
        # NOTE: not necessarily true e.g. for Linear layers
        # assert len(activations.shape) == 4
        return activations[
            self.batch_index[0] : self.batch_index[1], self.channel_index, ...
        ]


@loss_wrapper
class NeuronActivation(BaseLoss):
    def __init__(
        self,
        target: nn.Module,
        channel_index: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
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
        return activations[
            self.batch_index[0] : self.batch_index[1],
            self.channel_index,
            _x : _x + 1,
            _y : _y + 1,
        ]


@loss_wrapper
class DeepDream(BaseLoss):
    """
    Maximize 'interestingness' at the target layer.
    Mordvintsev et al., 2015.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return activations ** 2


@loss_wrapper
class TotalVariation(BaseLoss):
    """
    Total variation denoising penalty for activations.
    See Mahendran, V. 2014. Understanding Deep Image Representations by Inverting Them.
    https://arxiv.org/abs/1412.0035
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        x_diff = activations[..., 1:, :] - activations[..., :-1, :]
        y_diff = activations[..., :, 1:] - activations[..., :, :-1]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


@loss_wrapper
class L1(BaseLoss):
    """
    L1 norm of the target layer, generally used as a penalty.
    """

    def __init__(
        self,
        target: nn.Module,
        constant: float = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.constant = constant

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return torch.abs(activations - self.constant).sum()


@loss_wrapper
class L2(BaseLoss):
    """
    L2 norm of the target layer, generally used as a penalty.
    """

    def __init__(
        self,
        target: nn.Module,
        constant: float = 0.0,
        epsilon: float = 1e-6,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.constant = constant
        self.epsilon = epsilon

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target][
            self.batch_index[0] : self.batch_index[1]
        ]
        activations = ((activations - self.constant) ** 2).sum()
        return torch.sqrt(self.epsilon + activations)


@loss_wrapper
class Diversity(BaseLoss):
    """
    Use a cosine similarity penalty to extract features from a polysemantic neuron.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#diversity
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        batch, channels = activations.shape[:2]
        flattened = activations.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = nn.functional.normalize(grams, p=2, dim=(1, 2))
        return (
            -sum(
                [
                    sum([(grams[i] * grams[j]).sum() for j in range(batch) if j != i])
                    for i in range(batch)
                ]
            )
            / batch
        )


@loss_wrapper
class ActivationInterpolation(BaseLoss):
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
        self.target_one = target1
        self.channel_index_one = channel_index1
        self.target_two = target2
        self.channel_index_two = channel_index2
        # Expose targets for InputOptimization
        BaseLoss.__init__(self, [target1, target2])

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


@loss_wrapper
class Alignment(BaseLoss):
    """
    Penalize the L2 distance between tensors in the batch to encourage visual
    similarity between them.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons
    """

    def __init__(self, target: nn.Module, decay_ratio: float = 2.0) -> None:
        BaseLoss.__init__(self, target)
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

        return -sum_tensor


@loss_wrapper
class Direction(BaseLoss):
    """
    Visualize a general direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        cossim_pow: Optional[float] = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.direction = vec.reshape((1, -1, 1, 1))
        self.cossim_pow = cossim_pow

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations.size(1) == self.direction.size(1)
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return _dot_cossim(self.direction, activations, cossim_pow=self.cossim_pow)


@loss_wrapper
class NeuronDirection(BaseLoss):
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
        cossim_pow: Optional[float] = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.direction = vec.reshape((1, -1, 1, 1))
        self.x = x
        self.y = y
        self.channel_index = channel_index
        self.cossim_pow = cossim_pow

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        _x, _y = get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )
        activations = activations[
            self.batch_index[0] : self.batch_index[1], :, _x : _x + 1, _y : _y + 1
        ]
        if self.channel_index is not None:
            activations = activations[:, self.channel_index, ...][:, None, ...]
        return _dot_cossim(self.direction, activations, cossim_pow=self.cossim_pow)


@loss_wrapper
class AngledNeuronDirection(BaseLoss):
    """
    Visualize a direction vector with an optional whitened activation vector to
    unstretch the activation space. Compared to the traditional Direction objectives,
    this objective places more emphasis on angle by optionally multiplying the dot
    product by the cosine similarity.

    When cossim_pow is equal to 0, this objective works as a euclidean
    neuron objective. When cossim_pow is greater than 0, this objective works as a
    cosine similarity objective. An additional whitened neuron direction vector
    can optionally be supplied to improve visualization quality for some models.

    More information on the algorithm this objective uses can be found here:
    https://github.com/tensorflow/lucid/issues/116

    This Lucid equivalents of this loss function can be found here:
    https://github.com/tensorflow/lucid/blob/master/notebooks/
    activation-atlas/activation-atlas-simple.ipynb
    https://github.com/tensorflow/lucid/blob/master/notebooks/
    activation-atlas/class-activation-atlas.ipynb

    Like the Lucid equivalents, our implementation differs slightly from the
    associated research paper.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/
    """

    def __init__(
        self,
        target: torch.nn.Module,
        vec: torch.Tensor,
        vec_whitened: Optional[torch.Tensor] = None,
        cossim_pow: float = 4.0,
        x: Optional[int] = None,
        y: Optional[int] = None,
        eps: float = 1.0e-4,
        batch_index: Optional[int] = None,
    ) -> None:
        """
        Args:
            target (nn.Module): A target layer instance.
            vec (torch.Tensor): A neuron direction vector to use.
            vec_whitened (torch.Tensor, optional): A whitened neuron direction vector.
            cossim_pow (float, optional): The desired cosine similarity power to use.
            x (int, optional): Optionally provide a specific x position for the target
                neuron.
            y (int, optional): Optionally provide a specific y position for the target
                neuron.
            eps (float, optional): If cossim_pow is greater than zero, the desired
                epsilon value to use for cosine similarity calculations.
        """
        BaseLoss.__init__(self, target, batch_index)
        self.vec = vec.unsqueeze(0) if vec.dim() == 1 else vec
        self.vec_whitened = vec_whitened
        self.cossim_pow = cossim_pow
        self.eps = eps
        self.x = x
        self.y = y
        if self.vec_whitened is not None:
            assert self.vec_whitened.dim() == 2
        assert self.vec.dim() == 2

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        assert activations.dim() == 4 or activations.dim() == 2
        assert activations.shape[1] == self.vec.shape[1]
        if activations.dim() == 4:
            _x, _y = get_neuron_pos(
                activations.size(2), activations.size(3), self.x, self.y
            )
            activations = activations[..., _x, _y]

        vec = (
            torch.matmul(self.vec, self.vec_whitened)[0]
            if self.vec_whitened is not None
            else self.vec
        )
        if self.cossim_pow == 0:
            return activations * vec

        dot = torch.mean(activations * vec)
        cossims = dot / (self.eps + torch.sqrt(torch.sum(activations ** 2)))
        return dot * torch.clamp(cossims, min=0.1) ** self.cossim_pow


@loss_wrapper
class TensorDirection(BaseLoss):
    """
    Visualize a tensor direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        cossim_pow: Optional[float] = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        BaseLoss.__init__(self, target, batch_index)
        self.direction = vec
        self.cossim_pow = cossim_pow

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        H_direction, W_direction = self.direction.size(2), self.direction.size(3)
        H_activ, W_activ = activations.size(2), activations.size(3)

        H = (H_activ - H_direction) // 2
        W = (W_activ - W_direction) // 2

        activations = activations[
            self.batch_index[0] : self.batch_index[1],
            :,
            H : H + H_direction,
            W : W + W_direction,
        ]
        return _dot_cossim(self.direction, activations, cossim_pow=self.cossim_pow)


@loss_wrapper
class ActivationWeights(BaseLoss):
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
        BaseLoss.__init__(self, target)
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


def sum_loss_list(
    loss_list: List,
    to_scalar_fn: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> CompositeLoss:
    """
    Summarize a large number of losses without recursion errors. By default using 300+
    loss functions for a single optimization task will result in exceeding Python's
    default maximum recursion depth limit. This function can be used to avoid the
    recursion depth limit for tasks such as summarizing a large list of loss functions
    with the built-in sum() function.

    This function works similar to Lucid's optvis.objectives.Objective.sum() function.

    Args:

        loss_list (list): A list of loss function objectives.
        to_scalar_fn (Callable): A function for converting loss function outputs to
            scalar values, in order to prevent size mismatches.
            Default: torch.mean

    Returns:
        loss_fn (CompositeLoss): A composite loss function containing all the loss
            functions from `loss_list`.
    """

    def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
        return sum([to_scalar_fn(loss(module)) for loss in loss_list])

    name = "Sum(" + ", ".join([loss.__name__ for loss in loss_list]) + ")"
    # Collect targets from losses
    target = [
        target
        for targets in [
            [loss.target] if not isinstance(loss.target, list) else loss.target
            for loss in loss_list
        ]
        for target in targets
    ]

    # Filter out duplicate targets
    target = list(dict.fromkeys(target))
    return CompositeLoss(loss_fn, name=name, target=target)


def default_loss_summarize(loss_value: torch.Tensor) -> torch.Tensor:
    """
    Helper function to summarize tensor outputs from loss functions.

    default_loss_summarize applies `mean` to the loss tensor
    and negates it so that optimizing it maximizes the activations we
    are interested in.
    """
    return -1 * loss_value.mean()


__all__ = [
    "Loss",
    "loss_wrapper",
    "BaseLoss",
    "LayerActivation",
    "ChannelActivation",
    "NeuronActivation",
    "DeepDream",
    "TotalVariation",
    "L1",
    "L2",
    "Diversity",
    "ActivationInterpolation",
    "Alignment",
    "Direction",
    "NeuronDirection",
    "AngledNeuronDirection",
    "TensorDirection",
    "ActivationWeights",
    "sum_loss_list",
    "default_loss_summarize",
]
