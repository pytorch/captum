#!/usr/bin/env python3
import unittest
from typing import Union

import numpy as np
import torch

from captum.optim._utils.reducer import ChannelReducer
from tests.helpers.basic import BaseTest


class TestReductionAlgorithm(object):
    """
    Fake reduction algorithm for testing
    """

    def __init__(self, n_components=3, **kwargs) -> None:
        self.n_components = n_components

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return x[..., 0:3].numpy()


class TestChannelReducer(BaseTest):
    def test_channelreducer_pytorch(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch reshape test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.size(1), 3)

    def test_channelreducer_pytorch_pca(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch reshape test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = ChannelReducer(n_components=3, reduction_alg="PCA", max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.size(1), 3)

    def test_channelreducer_pytorch_custom_alg(self) -> None:
        test_input = torch.randn(1, 32, 224, 224).abs()
        reduction_alg = TestReductionAlgorithm
        c_reducer = ChannelReducer(
            n_components=3, reduction_alg=reduction_alg, max_iter=100
        )
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.size(1), 3)

    def test_channelreducer_numpy(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " NumPy reshape test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs().numpy()
        c_reducer = ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.shape[1], 3)

    def test_channelreducer_noreshape_pytorch(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch no reshape test"
            )

        test_input = torch.randn(1, 224, 224, 32).abs()
        c_reducer = ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=False)
        self.assertEquals(test_output.size(3), 3)

    def test_channelreducer_noreshape_numpy(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " NumPy no reshape test"
            )

        test_input = torch.randn(1, 224, 224, 32).abs().numpy()
        c_reducer = ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=False)
        self.assertEquals(test_output.shape[3], 3)


if __name__ == "__main__":
    unittest.main()
