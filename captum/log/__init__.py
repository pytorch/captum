#!/usr/bin/env python3

# pyre-strict

try:
    from captum.log.fb.internal_log import (
        disable_detailed_logging,
        log,
        log_usage,
        patch_methods,
        set_environment,
        TimedLog,
    )

    __all__ = [
        "log",
        "log_usage",
        "TimedLog",
        "set_environment",
        "disable_detailed_logging",
    ]

except ImportError:
    from functools import wraps

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def log(*args, **kwargs):
        pass

    # bug with mypy: https://github.com/python/mypy/issues/1153
    class TimedLog:  # type: ignore
        # pyre-fixme[2]: Parameter must be annotated.
        def __init__(self, *args, **kwargs) -> None:
            pass

        # pyre-fixme[3]: Return type must be annotated.
        def __enter__(self):
            return self

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def __exit__(self, exception_type, exception_value, traceback):
            return exception_value is not None

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def log_usage(*log_args, **log_kwargs):
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _log_usage(func):
            @wraps(func)
            # pyre-fixme[53]: Captured variable `func` is not annotated.
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return _log_usage

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def set_environment(env):
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def disable_detailed_logging():
        pass

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def patch_methods(tester, patch_log=True):
        pass
