#!/usr/bin/env python3

# pyre-strict

try:
    from captum.testing.helpers.fb.internal_base import FbBaseTest as BaseTest

    __all__ = [
        "BaseTest",
    ]

except ImportError:
    # tests/helpers/__init__.py:13: error: Incompatible import of "BaseTest"
    # (imported name has type "type[BaseTest]", local name has type
    # "type[FbBaseTest]")  [assignment]
    from captum.testing.helpers.basic import BaseTest  # type: ignore
