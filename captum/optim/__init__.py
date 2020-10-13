from typing import Dict, Optional, Union, Callable, Iterable
from typing_extensions import Protocol

import torch
import torch.nn as nn

ParametersForOptimizers = Iterable[Union[torch.Tensor, Dict[str, torch.tensor]]]


class HasLoss(Protocol):
    def loss(self) -> torch.Tensor:
        ...


class Parameterized(Protocol):
    parameters: ParametersForOptimizers


class Objective(Parameterized, HasLoss):
    def cleanup(self):
        pass


ModuleOutputMapping = Dict[nn.Module, Optional[torch.Tensor]]

StopCriteria = Callable[[int, Objective, torch.optim.Optimizer], bool]

