#!/usr/bin/env python3

# pyre-strict

import typing
from types import TracebackType
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    Iterator,
    Literal,
    Optional,
    TextIO,
    Type,
    TypeVar,
    Union,
)

from tqdm.auto import tqdm

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


@typing.overload
def progress(
    iterable: None = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> tqdm: ...


@typing.overload
def progress(
    iterable: Iterable[IterableType],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> tqdm: ...


def progress(
    iterable: Optional[Iterable[IterableType]] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> tqdm:
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        file=file,
        mininterval=mininterval,
        **kwargs,
    )
