from __future__ import print_function

import torch

from captum.attr._utils.common import validate_input, validate_noise_tunnel_type
from captum.attr._utils.common import Stat, MaxList
from captum.attr._core.noise_tunnel import SUPPORTED_NOISE_TUNNEL_TYPES

from .helpers.utils import BaseTest


class Test(BaseTest):
    def test_validate_input(self):
        with self.assertRaises(AssertionError):
            validate_input(torch.Tensor([-1.0, 1.0]), torch.Tensor([-2.0]))
            validate_input(
                torch.Tensor([-1.0, 1.0]), torch.Tensor([-1.0, 1.0]), n_steps=-1
            )
            validate_input(
                torch.Tensor([-1.0, 1.0]), torch.Tensor([-1.0, 1.0]), method="abcde"
            )
        validate_input(torch.Tensor([-1.0]), torch.Tensor([-2.0]))
        validate_input(
            torch.Tensor([-1.0]), torch.Tensor([-2.0]), method="gausslegendre"
        )

    def test_validate_nt_type(self):
        with self.assertRaises(AssertionError):
            validate_noise_tunnel_type("abc", SUPPORTED_NOISE_TUNNEL_TYPES)
        validate_noise_tunnel_type("smoothgrad", SUPPORTED_NOISE_TUNNEL_TYPES)
        validate_noise_tunnel_type("smoothgrad_sq", SUPPORTED_NOISE_TUNNEL_TYPES)
        validate_noise_tunnel_type("vargrad", SUPPORTED_NOISE_TUNNEL_TYPES)

    def test_stat_tracking(self):
        data = [1, 2, 3, 4, 5]
        s = Stat()
        s.update(data)

        self.assertEqual(s.get_mean(), 3.0)
        self.assertEqual(s.get_std(), 2.0 ** 0.5)
        self.assertEqual(s.get_variance(), 2.0)
        self.assertEqual(s.get_sample_variance(), 10.0 / 4)
        self.assertEqual(s.get_min(), 1.0)
        self.assertEqual(s.get_max(), 5.0)
        self.assertEqual(s.get_count(), 5)
        self.assertEqual(
            s.get_stats(),
            {
                "mean": 3.0,
                "sample_variance": 10.0 / 4,
                "variance": 2.0,
                "std": 2.0 ** 0.5,
                "min": 1.0,
                "max": 5.0,
                "count": 5,
            },
        )
        s.update([0.34, 0.95])
        self.assertAlmostEqual(s.get_mean(), 2.3271428571429)
        self.assertEqual(s.get_count(), 7)

    def test_max_list(self):
        ml = MaxList(3)
        ml.add(5)
        ml.add(2)
        ml.add(1)

        self.assertEqual(ml.get_list(), [5, 2, 1])

        ml.add(3)
        ml.add(1)

        self.assertEqual(ml.get_list(), [5, 3, 2])

    def test_max_item(self):
        ml = MaxList(1)
        ml.add(5)
        ml.add(2)
        ml.add(8)
        ml.add(3)

        self.assertEqual(ml.get_list(), [8])

    def test_max_list_string(self):
        ml = MaxList(4, key=lambda x: len(x))
        ml.add("American Trade Deal")
        ml.add("Chryseler")
        ml.add("Supercalafragilisticexpialodocious")
        ml.add("ravenclaw")
        ml.add("Facebook Rocks!")

        self.assertEqual(ml.get_list()[3], "Facebook Rocks!")
        self.assertEqual(len(ml.get_list()), 4)
