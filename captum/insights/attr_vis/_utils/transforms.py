#!/usr/bin/env python3

# pyre-strict

from typing import Callable, List, Optional, Union


def format_transforms(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    transforms: Optional[Union[Callable, List[Callable]]]
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
) -> List[Callable]:
    if transforms is None:
        return []
    if callable(transforms):
        return [transforms]
    return transforms
