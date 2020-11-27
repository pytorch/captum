#!/usr/bin/env python3
import unittest

from captum.optim._utils.images import get_neuron_pos
from tests.helpers.basic import BaseTest


class TestGetNeuronPos(BaseTest):
    def test_get_neuron_pos(self) -> None:
        W, H = 128, 128
        x, y = get_neuron_pos(H, W)

        assert x == W // 2
        assert y == H // 2

        x, y = get_neuron_pos(H, W, 5, 5)

        assert x == 5
        assert y == 5


if __name__ == "__main__":
    unittest.main()
