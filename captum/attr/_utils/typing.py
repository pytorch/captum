#!/usr/bin/env python3

from typing import Tuple, TypeVar
from torch import Tensor

TensorOrTuple = TypeVar("TensorOrTuple", Tensor, Tuple[Tensor, ...])
