#!/usr/bin/env python3

from typing import Tuple, TypeVar, TYPE_CHECKING
from torch import Tensor

import sys

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal  # noqa: F401

TensorOrTupleOfTensors = TypeVar("TensorOrTupleOfTensors", Tensor, Tuple[Tensor, ...])
TupleOrTensorOrBool = TypeVar("TupleOrTensorOrBool", Tuple, Tensor, bool)
TrueOrFalse = TypeVar("TrueOrFalse", "Literal"[True], "Literal"[False])
