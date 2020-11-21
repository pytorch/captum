import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


ParametersForOptimizers = Iterable[Union[Tensor, Dict[str, Tensor]]]


class HasLoss(Protocol):
    def loss(self) -> Tensor:
        ...


class Parameterized(Protocol):
    parameters: ParametersForOptimizers


class Objective(Parameterized, HasLoss):
    def cleanup(self):
        pass


ModuleOutputMapping = Dict[Module, Optional[Tensor]]
StopCriteria = Callable[[int, Objective, Optimizer], bool]
LossFunction = Callable[[ModuleOutputMapping], Tensor]
SingleTargetLossFunction = Callable[[Tensor], Tensor]

InitSize = Tuple[int, int]
SquashFunc = Callable[[Tensor], Tensor]
TransformValList = Union[Sequence[int], Sequence[float], Tensor]
TransformVal = Union[int, float, Tensor]
TransformSize = Union[List[int, int], Tuple[int, int], int]
