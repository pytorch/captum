#!/usr/bin/env python3

from typing import Tuple, TypeVar
from torch import Tensor

TensorOrTupleOfTensors = TypeVar("TensorOrTupleOfTensors", Tensor, Tuple[Tensor, ...])
