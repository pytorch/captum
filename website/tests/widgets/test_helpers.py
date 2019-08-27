#!/usr/bin/env python3
import unittest

from captum.widgets.visualization_helpers import _multiple_item_selection


class Test(unittest.TestCase):
    def test_multiple_item_selection(self):
        _multiple_item_selection(["cat1", "cat2", "cat3"])


if __name__ == "__main__":
    unittest.main()
