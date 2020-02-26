#!/usr/bin/env python3

from typing import Callable, List, Optional, Union


def format_transforms(
    transforms: Optional[Union[Callable, List[Callable]]]
) -> List[Callable]:
    if transforms is None:
        return []
    if callable(transforms):
        return [transforms]
    return transforms
