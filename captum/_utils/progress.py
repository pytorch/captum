#!/usr/bin/env python3

# pyre-strict

import sys
import typing
import warnings
from time import time
from types import TracebackType
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sized,
    TextIO,
    Type,
    TypeVar,
    Union,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

T = TypeVar("T")
IterableType = TypeVar("IterableType")


class DisableErrorIOWrapper(object):
    def __init__(self, wrapped: TextIO) -> None:
        """
        The wrapper around a TextIO object to ignore write errors like tqdm
        https://github.com/tqdm/tqdm/blob/bcce20f771a16cb8e4ac5cc5b2307374a2c0e535/tqdm/utils.py#L131
        """
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)

    @staticmethod
    def _wrapped_run(
        func: Callable[..., T], *args: object, **kwargs: object
    ) -> Union[T, None]:
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno != 5:
                raise
        except ValueError as e:
            if "closed" not in str(e):
                raise
        return None

    def write(self, *args: object, **kwargs: object) -> Optional[int]:
        return self._wrapped_run(self._wrapped.write, *args, **kwargs)

    def flush(self, *args: object, **kwargs: object) -> None:
        return self._wrapped_run(self._wrapped.flush, *args, **kwargs)


class NullProgress(Iterable[IterableType]):
    """Passthrough class that implements the progress API.

    This class implements the tqdm and SimpleProgressBar api but
    does nothing. This class can be used as a stand-in for an
    optional progressbar, most commonly in the case of nested
    progress bars.
    """

    def __init__(
        self,
        iterable: Optional[Iterable[IterableType]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.iterable = iterable

    def __enter__(self) -> "NullProgress[IterableType]":
        return self

    def __exit__(
        self,
        exc_type: Union[Type[BaseException], None],
        exc_value: Union[BaseException, None],
        exc_traceback: Union[TracebackType, None],
    ) -> Literal[False]:
        return False

    def __iter__(self) -> Iterator[IterableType]:
        if not self.iterable:
            return
        for it in cast(Iterable[IterableType], self.iterable):
            yield it

    def update(self, amount: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


class SimpleProgress(Iterable[IterableType]):
    def __init__(
        self,
        iterable: Optional[Iterable[IterableType]] = None,
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

        file_wrapper = DisableErrorIOWrapper(file if file else sys.stderr)
        self.file: DisableErrorIOWrapper = file_wrapper

        self.mininterval = mininterval
        self.last_print_t = 0.0
        self.closed = False
        self._is_parent = False

    def __enter__(self) -> "SimpleProgress[IterableType]":
        self._is_parent = True
        self._refresh()
        return self

    def __exit__(
        self,
        exc_type: Union[Type[BaseException], None],
        exc_value: Union[BaseException, None],
        exc_traceback: Union[TracebackType, None],
    ) -> Literal[False]:
        self.close()
        return False

    def __iter__(self) -> Iterator[IterableType]:
        if self.closed or not self.iterable:
            return
        self._refresh()
        for it in cast(Iterable[IterableType], self.iterable):
            yield it
            self.update()
        self.close()

    def _refresh(self) -> None:
        progress_str = self.desc + ": " if self.desc else ""
        if self.total:
            # e.g., progress: 60% 3/5
            progress_str += (
                f"{100 * self.cur // cast(int, self.total)}%"
                f" {self.cur}/{cast(int, self.total)}"
            )
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


@typing.overload
def progress(
    iterable: None = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    use_tqdm: bool = True,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> Union[SimpleProgress[None], tqdm]: ...


@typing.overload
def progress(
    iterable: Iterable[IterableType],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    use_tqdm: bool = True,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> Union[SimpleProgress[IterableType], tqdm]: ...


def progress(
    iterable: Optional[Iterable[IterableType]] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    use_tqdm: bool = True,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> Union[SimpleProgress[IterableType], tqdm]:
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
