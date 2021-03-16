#!/usr/bin/env python3

import sys
import warnings
from time import time
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


class SimpleProgress:
    def __init__(
        self,
        iterable: Iterable = None,
        desc: str = None,
        total: int = None,
        file: TextIO = None,
        mininterval: float = 0.5,
    ):
        """
        Simple progress output used when tqdm is unavailable.
        Same as tqdm, output to stderr channel
        """
        self.cur = 0

        self.iterable = iterable
        self.total = total
        if total is None and hasattr(iterable, "__len__"):
            self.total = len(cast(Sized, iterable))

        self.desc = desc

        file = DisableErrorIOWrapper(file if file else sys.stderr)
        cast(TextIO, file)
        self.file = file

        self.mininterval = mininterval
        self.last_print_t = 0.0
        self.closed = False

    def __iter__(self):
        if self.closed or not self.iterable:
            return
        self._refresh()
        for it in self.iterable:
            yield it
            self.update()
        self.close()

    def _refresh(self):
        progress_str = self.desc + ": " if self.desc else ""
        if self.total:
            # e.g., progress: 60% 3/5
            progress_str += f"{100 * self.cur // self.total}% {self.cur}/{self.total}"
        else:
            # e.g., progress: .....
            progress_str += "." * self.cur

        print("\r" + progress_str, end="", file=self.file)

    def update(self, amount: int = 1):
        if self.closed:
            return
        self.cur += amount

        cur_t = time()
        if cur_t - self.last_print_t >= self.mininterval:
            self._refresh()
            self.last_print_t = cur_t

    def close(self):
        if not self.closed:
            self._refresh()
            print(file=self.file)  # end with new line
            self.closed = True


def progress(
    iterable: Iterable = None,
    desc: str = None,
    total: int = None,
    use_tqdm=True,
    file: TextIO = None,
    mininterval: float = 0.5,
    **kwargs,
):
    # Try to use tqdm is possible. Fall back to simple progress print
    if tqdm and use_tqdm:
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            file=file,
            mininterval=mininterval,
            **kwargs,
        )
    else:
        if not tqdm and use_tqdm:
            warnings.warn(
                "Tried to show progress with tqdm "
                "but tqdm is not installed. "
                "Fall back to simply print out the progress."
            )
        return SimpleProgress(
            iterable, desc=desc, total=total, file=file, mininterval=mininterval
        )
