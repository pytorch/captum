#!/usr/bin/env python3

import sys
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _simple_progress_out(iterable: Iterable, desc: str = None, total: int = None):
    """
    Simple progress output used when tqdm is unavailable.
    Same as tqdm, output to stderr channel
    """
    cur = 0

    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)

    desc = desc + ": " if desc else ""
    progress_str = (
        lambda cur: f"{desc}{100 * cur // total}% {cur}/{total}"
        if total
        else f"{desc}{'.' * cur}"
    )

    print("\r" + progress_str(cur), end="", file=sys.stderr)
    for it in iterable:
        yield it
        cur += 1
        print("\r" + progress_str(cur), end="", file=sys.stderr)

    print(file=sys.stderr)  # end with new line


def progress(
    iterable: Iterable, desc: str = None, total: int = None, use_tqdm=True, **kwargs
):
    # Try to use tqdm is possible. Fall back to simple progress print
    if tqdm and use_tqdm:
        return tqdm(iterable, desc=desc, total=total, **kwargs)
    else:
        return _simple_progress_out(iterable, desc=desc, total=total)
