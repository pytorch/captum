#!/usr/bin/env python3

# pyre-strict

from typing import Any, Dict, Optional, Protocol, Tuple, Type

import torch

from packaging.version import Version
from torch import nn


class CacheLike(Protocol):
    """Protocol for cache-like objects."""


class DynamicCacheLike(CacheLike, Protocol):
    """Protocol for dynamic cache-like objects."""

    @classmethod
    def from_legacy_cache(
        cls: Type["DynamicCacheLike"],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> "DynamicCacheLike": ...


transformers_installed: bool
Cache: Optional[Type[CacheLike]]
DynamicCache: Optional[Type[DynamicCacheLike]]

try:
    # pyre-ignore[21]: Could not find a module corresponding to import `transformers`.
    import transformers  # noqa: F401

    transformers_installed = True
except ImportError:
    transformers_installed = False

if transformers_installed:
    try:
        # pyre-ignore[21]: Could not find a module corresponding to import
        # `transformers.cache_utils`.
        from transformers.cache_utils import (  # noqa: F401
            Cache as _Cache,
            DynamicCache as _DynamicCache,
        )

        Cache = _Cache
        # pyre-ignore[9]: Incompatible variable type: DynamicCache is declared to have
        # type `Optional[Type[DynamicCacheLike]]` but is used as type
        # `Type[_DynamicCache]`
        DynamicCache = _DynamicCache
    except ImportError:
        Cache = DynamicCache = None
else:
    Cache = DynamicCache = None

# GenerationMixin._update_model_kwargs_for_generation
# "cache_position" at v4.39.0 (only needed for models that support cache class)
# "use_cache" at v4.41.0 (optional, default is True)
# "cache_position" is mandatory at v4.43.0 ("use_cache" is still optional, default True)
_transformers_version: Optional[Version]
if transformers_installed:
    _transformers_version = Version(transformers.__version__)
else:
    _transformers_version = None

_mandated_cache_version = Version("4.43.0")
_use_cache_version = Version("4.41.0")
_cache_position_version = Version("4.39.0")


def update_model_kwargs(
    model_kwargs: Dict[str, Any],
    model: nn.Module,
    input_ids: torch.Tensor,
    caching: bool,
) -> None:
    if not supports_caching(model):
        return
    assert _transformers_version is not None
    if caching:
        # Enable caching
        if _transformers_version >= _cache_position_version:
            cache_position = torch.arange(
                input_ids.shape[1], dtype=torch.int64, device=input_ids.device
            )
            model_kwargs["cache_position"] = cache_position
        # pyre-ignore[58]: Unsupported operand `>=` is not supported for operand types
        # `Optional[Version]` and `Version`.
        if _transformers_version >= _use_cache_version:
            model_kwargs["use_cache"] = True
    else:
        # Disable caching
        if _transformers_version >= _use_cache_version:
            model_kwargs["use_cache"] = False


def supports_caching(model: nn.Module) -> bool:
    if not transformers_installed:
        # Not a transformers model
        return False
    # Cache may be optional or unsupported depending on model/version
    try:
        # pyre-ignore[21]: Could not find a module corresponding to import
        # `transformers.generation.utils`.
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return False
    if not isinstance(model, GenerationMixin):
        # Model isn't a GenerationMixin, we don't support additional caching logic
        # for it
        return False
    assert _transformers_version is not None
    if _transformers_version >= _mandated_cache_version:
        # Cache is mandatory
        return True
    # Fallback on _supports_cache_class attribute
    # pyre-fixme[7]: Expected `bool` but got `Union[Module, Tensor]`.
    return getattr(model, "_supports_cache_class", False)
