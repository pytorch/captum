#!/usr/bin/env python3

# pyre-strict

from collections import UserDict
from typing import (
    List,
    Literal,
    Optional,
    overload,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from torch import Tensor
from torch.nn import Module

TensorOrTupleOfTensorsGeneric = TypeVar(
    "TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...]
)
# pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
TupleOrTensorOrBoolGeneric = TypeVar("TupleOrTensorOrBoolGeneric", Tuple, Tensor, bool)
ModuleOrModuleList = TypeVar("ModuleOrModuleList", Module, List[Module])
TargetType = Union[None, int, Tuple[int, ...], Tensor, List[Tuple[int, ...]], List[int]]
BaselineTupleType = Union[None, Tuple[Union[Tensor, int, float], ...]]
BaselineType = Union[None, Tensor, int, float, BaselineTupleType]

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


# Necessary for Python >=3.7 and <3.9!
if TYPE_CHECKING:
    BatchEncodingType = UserDict[Union[int, str], object]
else:
    BatchEncodingType = UserDict


class TokenizerLike(Protocol):
    """A protocol for tokenizer-like objects that can be used with Captum
    LLM attribution methods."""

    @overload
    def encode(
        self, text: str, add_special_tokens: bool = ..., return_tensors: None = ...
    ) -> List[int]: ...

    @overload
    def encode(
        self,
        text: str,
        add_special_tokens: bool = ...,
        return_tensors: Literal["pt"] = ...,
    ) -> Tensor: ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Tensor]: ...

    def decode(self, token_ids: Tensor) -> str: ...

    @overload
    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]: ...
    @overload
    def convert_ids_to_tokens(self, token_ids: int) -> str: ...

    def convert_ids_to_tokens(
        self, token_ids: Union[List[int], int]
    ) -> Union[List[str], str]: ...

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...
    @overload
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]: ...

    def convert_tokens_to_ids(
        self, tokens: Union[List[str], str]
    ) -> Union[List[int], int]: ...

    def __call__(
        self,
        text: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> BatchEncodingType: ...
