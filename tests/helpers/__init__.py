#!/usr/bin/env python3

# pyre-strict

try:
    from tests.helpers.fb.internal_base import FbBaseTest as BaseTest

    __all__ = [
        "BaseTest",
    ]

except ImportError:
    from tests.helpers.basic import BaseTest
