#!/usr/bin/env python3

# pyre-strict

import sys
import warnings
from time import time
from typing import Any, cast, Iterable, Literal, Optional, Sized, TextIO

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class DisableErrorIOWrapper(object):
    def __init__(self, wrapped: TextIO) -> None:
        """
        The wrapper around a TextIO object to ignore write errors like tqdm
        https://github.com/tqdm/tqdm/blob/bcce20f771a16cb8e4ac5cc5b2307374a2c0e535/tqdm/utils.py#L131
        """
        self._wrapped = wrapped

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _wrapped_run(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno != 5:
                raise
        except ValueError as e:
            if "closed" not in str(e):
                raise

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def write(self, *args, **kwargs):
        return self._wrapped_run(self._wrapped.write, *args, **kwargs)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def flush(self, *args, **kwargs):
        return self._wrapped_run(self._wrapped.flush, *args, **kwargs)


class NullProgress:
    """Passthrough class that implements the progress API.

    This class implements the tqdm and SimpleProgressBar api but
    does nothing. This class can be used as a stand-in for an
    optional progressbar, most commonly in the case of nested
    progress bars.
    """

    def __init__(
        self,
        # pyre-fixme[24]: Generic type `Iterable` expects 1 type parameter.
        iterable: Optional[Iterable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.iterable = iterable

    def __enter__(self) -> "NullProgress":
        return self

    # pyre-fixme[2]: Parameter must be annotated.
    def __exit__(self, exc_type, exc_value, exc_traceback) -> Literal[False]:
        return False

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        if not self.iterable:
            return
        # pyre-fixme[16]: `Optional` has no attribute `__iter__`.
        for it in self.iterable:
            yield it

    def update(self, amount: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


class SimpleProgress:
    def __init__(
        self,
        # pyre-fixme[24]: Generic type `Iterable` expects 1 type parameter.
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        file: Optional[TextIO] = None,
        mininterval: float = 0.5,
    ) -> None:
        """
        Simple progress output used when tqdm is unavailable.
        Same as tqdm, output to stderr channel.
        If you want to do nested Progressbars with simple progress
        the parent progress bar should be used as a context
        (i.e. with statement) and the nested progress bar should be
        created inside this context.
        """
        self.cur = 0
        self.iterable = iterable
        self.total = total
        if total is None and hasattr(iterable, "__len__"):
            self.total = len(cast(Sized, iterable))

        self.desc = desc

        # pyre-fixme[9]: file has type `Optional[TextIO]`; used as
        #  `DisableErrorIOWrapper`.
        file = DisableErrorIOWrapper(file if file else sys.stderr)
        cast(TextIO, file)
        self.file = file

        self.mininterval = mininterval
        self.last_print_t = 0.0
        self.closed = False
        self._is_parent = False

    def __enter__(self) -> "SimpleProgress":
        self._is_parent = True
        self._refresh()
        return self

    # pyre-fixme[2]: Parameter must be annotated.
    def __exit__(self, exc_type, exc_value, exc_traceback) -> Literal[False]:
        self.close()
        return False

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        if self.closed or not self.iterable:
            return
        self._refresh()
        # pyre-fixme[16]: `Optional` has no attribute `__iter__`.
        for it in self.iterable:
            yield it
            self.update()
        self.close()

    def _refresh(self) -> None:
        progress_str = self.desc + ": " if self.desc else ""
        if self.total:
            # e.g., progress: 60% 3/5
            # pyre-fixme[58]: `//` is not supported for operand types `int` and
            #  `Optional[int]`.
            progress_str += f"{100 * self.cur // self.total}% {self.cur}/{self.total}"
        else:
            # e.g., progress: .....
            progress_str += "." * self.cur
        end = "\n" if self._is_parent else ""
        print("\r" + progress_str, end=end, file=self.file)

    def update(self, amount: int = 1) -> None:
        if self.closed:
            return
        self.cur += amount

        cur_t = time()
        if cur_t - self.last_print_t >= self.mininterval:
            self._refresh()
            self.last_print_t = cur_t

    def close(self) -> None:
        if not self.closed and not self._is_parent:
            self._refresh()
            print(file=self.file)  # end with new line
            self.closed = True


# pyre-fixme[3]: Return type must be annotated.
def progress(
    # pyre-fixme[24]: Generic type `Iterable` expects 1 type parameter.
    iterable: Optional[Iterable] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    use_tqdm: bool = True,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    # pyre-fixme[2]: Parameter must be annotated.
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
                "Fall back to simply print out the progress.",
                stacklevel=1,
            )
        return SimpleProgress(
            iterable, desc=desc, total=total, file=file, mininterval=mininterval
        )
