#!/usr/bin/env python3
import unittest

from captum.optim._utils.images import get_neuron_pos


class TestGetNeuronPos(unittest.TestCase):
    def test_get_neuron_pos(self) -> None:
        W, H = 128, 128
        x, y = get_neuron_pos(H, W)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, H // 2)

        x, y = get_neuron_pos(H, W, 5, 5)

        self.assertEqual(x, 5)
        self.assertEqual(y, 5)

        x, y = get_neuron_pos(H, W, 5, None)

        self.assertEqual(x, 5)
        self.assertEqual(y, H // 2)

        x, y = get_neuron_pos(H, W, None, 5)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, 5)


if __name__ == "__main__":
    unittest.main()
