#!/usr/bin/env python3

# pyre-strict

from typing import Optional, Protocol, Tuple, Type

import torch


class CacheLike(Protocol):
    """Protocol for cache-like objects."""


class DynamicCacheLike(CacheLike, Protocol):
    """Protocol for dynamic cache-like objects."""

    @classmethod
    def from_legacy_cache(
        cls: Type["DynamicCacheLike"],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> "DynamicCacheLike": ...


try:
    # pyre-ignore[21]: Could not find a module corresponding to import
    #  `transformers.cache_utils`
    from transformers.cache_utils import Cache as _Cache, DynamicCache as _DynamicCache
except ImportError:
    _Cache = _DynamicCache = None

Cache: Optional[Type[CacheLike]] = _Cache
DynamicCache: Optional[Type[DynamicCacheLike]] = _DynamicCache
