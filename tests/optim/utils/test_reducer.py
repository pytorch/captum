#!/usr/bin/env python3
import unittest
from typing import Union

import numpy as np
import torch

import captum.optim._utils.reducer as reducer
from tests.helpers.basic import BaseTest


class FakeReductionAlgorithm(object):
    """
    Fake reduction algorithm for testing
    """

    def __init__(self, n_components=3, **kwargs) -> None:
        self.n_components = n_components
        self.components_ = np.ones((2, 64))

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        x = x.numpy() if torch.is_tensor(x) else x
        return x[..., 0:3]


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
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.size(1), 3)

    def test_channelreducer_pytorch_pca(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch reshape PCA test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = reducer.ChannelReducer(
            n_components=3, reduction_alg="PCA", max_iter=100
        )
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.size(1), 3)

    def test_channelreducer_pytorch_custom_alg(self) -> None:
        test_input = torch.randn(1, 32, 224, 224).abs()
        reduction_alg = FakeReductionAlgorithm
        c_reducer = reducer.ChannelReducer(
            n_components=3, reduction_alg=reduction_alg, max_iter=100
        )
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        self.assertEquals(test_output.size(1), 3)

    def test_channelreducer_pytorch_custom_alg_components(self) -> None:
        reduction_alg = FakeReductionAlgorithm
        c_reducer = reducer.ChannelReducer(
            n_components=3, reduction_alg=reduction_alg, max_iter=100
        )
        components = c_reducer.components
        self.assertTrue(torch.is_tensor(components))

    def test_channelreducer_pytorch_components(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch reshape test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=True)
        components = c_reducer.components
        self.assertTrue(torch.is_tensor(components))
        self.assertTrue(torch.is_tensor(test_output))

    def test_channelreducer_numpy(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " NumPy reshape test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs().numpy()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
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
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
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
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, reshape=False)
        self.assertEquals(test_output.shape[3], 3)


class TestPosNeg(BaseTest):
    def test_posneg(self) -> None:
        x = torch.ones(1, 3, 224, 224) - 2
        self.assertGreater(
            torch.sum(reducer.posneg(x) >= 0).item(), torch.sum(x >= 0).item()
        )


class TestNChannelsToRGB(BaseTest):
    def test_nchannels_to_rgb_collapse(self) -> None:
        test_input = torch.randn(1, 6, 224, 224)
        test_output = reducer.nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])

    def test_nchannels_to_rgb_increase(self) -> None:
        test_input = torch.randn(1, 2, 224, 224)
        test_output = reducer.nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])


if __name__ == "__main__":
    unittest.main()
