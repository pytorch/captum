#!/usr/bin/env python3

import sys
import warnings
from typing import Iterable, Sized, TextIO, cast

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class DisableErrorIOWrapper(object):
    def __init__(self, wrapped: TextIO):
        """
        The wrapper around a TextIO object to ignore write errors like tqdm
        https://github.com/tqdm/tqdm/blob/bcce20f771a16cb8e4ac5cc5b2307374a2c0e535/tqdm/utils.py#L131
        """
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    @staticmethod
    def _wrapped_run(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno != 5:
                raise
        except ValueError as e:
            if "closed" not in str(e):
                raise

    def write(self, *args, **kwargs):
        return self._wrapped_run(self._wrapped.write, *args, **kwargs)

    def flush(self, *args, **kwargs):
        return self._wrapped_run(self._wrapped.flush, *args, **kwargs)


def _simple_progress_out(
    iterable: Iterable, desc: str = None, total: int = None, file: TextIO = None
):
    """
    Simple progress output used when tqdm is unavailable.
    Same as tqdm, output to stderr channel
    """
    cur = 0

    if total is None and hasattr(iterable, "__len__"):
        total = len(cast(Sized, iterable))

    desc = desc + ": " if desc else ""

    def _progress_str(cur):
        if total:
            # e.g., progress: 60% 3/5
            return f"{desc}{100 * cur // total}% {cur}/{total}"
        else:
            # e.g., progress: .....
            return f"{desc}{'.' * cur}"

    if not file:
        file = sys.stderr
    file = DisableErrorIOWrapper(file)

    print("\r" + _progress_str(cur), end="", file=file)
    for it in iterable:
        yield it
        cur += 1
        print("\r" + _progress_str(cur), end="", file=file)

    print(file=file)  # end with new line


def progress(
    iterable: Iterable,
    desc: str = None,
    total: int = None,
    use_tqdm=True,
    file: TextIO = None,
    **kwargs,
):
    # Try to use tqdm is possible. Fall back to simple progress print
    if tqdm and use_tqdm:
        return tqdm(iterable, desc=desc, total=total, file=file, **kwargs)
    else:
        if not tqdm and use_tqdm:
            warnings.warn(
                "Tried to show progress with tqdm "
                "but tqdm is not installed. "
                "Fall back to simply print out the progress."
            )
        return _simple_progress_out(iterable, desc=desc, total=total, file=file)
