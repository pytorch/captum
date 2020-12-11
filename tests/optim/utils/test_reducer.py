#!/usr/bin/env python3
import unittest

import torch

from captum.optim._utils.reducer import ChannelReducer
from tests.helpers.basic import BaseTest


class TestChannelReducer(BaseTest):
    def test_channelreducer(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input)
        self.assertEquals(test_output.size(1), 3)


if __name__ == "__main__":
    unittest.main()
