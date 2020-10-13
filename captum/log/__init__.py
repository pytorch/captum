#!/usr/bin/env python3

try:
    from captum.log.fb.internal_log import (
        TimedLog,
        log,
        log_usage,
        patch_methods,
        set_environment,
    )

    __all__ = ["log", "log_usage", "TimedLog", "set_environment"]

except ImportError:
    from functools import wraps

    def log(*args, **kwargs):
        pass

    # bug with mypy: https://github.com/python/mypy/issues/1153
    class TimedLog:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            return exception_value is not None

    def log_usage(*log_args, **log_kwargs):
        def _log_usage(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return _log_usage

    def set_environment(env):
        pass

    def patch_methods(tester, patch_log=True):
        pass
