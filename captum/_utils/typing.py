#!/usr/bin/env python3

from typing import List, Tuple, TYPE_CHECKING, TypeVar, Union

from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 8):
        from typing import Literal  # noqa: F401
    else:
        from typing_extensions import Literal  # noqa: F401
else:
    Literal = {True: bool, False: bool, (True, False): bool}

TensorOrTupleOfTensorsGeneric = TypeVar(
    "TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...]
)
TupleOrTensorOrBoolGeneric = TypeVar("TupleOrTensorOrBoolGeneric", Tuple, Tensor, bool)
ModuleOrModuleList = TypeVar("ModuleOrModuleList", Module, List[Module])
TargetType = Union[None, int, Tuple[int, ...], Tensor, List[Tuple[int, ...]], List[int]]
BaselineType = Union[None, Tensor, int, float, Tuple[Union[Tensor, int, float], ...]]

TensorLikeList1D = List[float]
TensorLikeList2D = List[TensorLikeList1D]
TensorLikeList3D = List[TensorLikeList2D]
TensorLikeList4D = List[TensorLikeList3D]
TensorLikeList5D = List[TensorLikeList4D]
TensorLikeList = Union[
    TensorLikeList1D,
    TensorLikeList2D,
    TensorLikeList3D,
    TensorLikeList4D,
    TensorLikeList5D,
]
