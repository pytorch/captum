#!/usr/bin/env python3

# pyre-strict

from typing import List, Optional, Protocol, Tuple, TYPE_CHECKING, TypeVar, Union

from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from typing import Literal
else:
    Literal = {True: bool, False: bool, (True, False): bool}

TensorOrTupleOfTensorsGeneric = TypeVar(
    "TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...]
)
# pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
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


class TokenizerLike(Protocol):
    """A protocol for tokenizer-like objects that can be used with Captum
    LLM attribution methods."""

    def encode(
        self, text: str, return_tensors: Optional[str] = None
    ) -> Union[List[int], Tensor]: ...
    def decode(self, token_ids: Tensor) -> str: ...
    def convert_ids_to_tokens(self, token_ids: Tensor) -> List[str]: ...
