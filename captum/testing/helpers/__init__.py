#!/usr/bin/env python3

# pyre-strict

try:
    from captum.testing.helpers.fb.internal_base import (  # type: ignore
        FbBaseTest as _BaseTest,
    )

except ImportError:
    # tests/helpers/__init__.py:13: error: Incompatible import of "_BaseTest"
    # (imported name has type "type[BaseTest]", local name has type
    # "type[FbBaseTest]")  [assignment]
    from captum.testing.helpers.basic import BaseTest as _BaseTest  # type: ignore

BaseTest = _BaseTest

__all__ = [
    "BaseTest",
]
