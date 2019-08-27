from __future__ import print_function

import torch
import unittest

from captum.attributions.utils.common import validate_input, validate_reg_type
from captum.attributions.utils.common import maximum_of_lists
from captum.attributions.utils.common import normalize
from captum.attributions.utils.common import Stat, MaxList


class Test(unittest.TestCase):
    def test_maximum_of_lists(self):
        a = torch.Tensor([1.0, -3.0, -7.4])
        b = torch.Tensor([-2.0, 1.5])
        c = torch.Tensor([5.0, -6.0])

        self.assertEqual(-7.4, maximum_of_lists(a, b))
        self.assertEqual(7.4, maximum_of_lists(a, b, abs_val=True))
        self.assertEqual(-6.0, maximum_of_lists(b, c))
        self.assertEqual(6.0, maximum_of_lists(b, c, abs_val=True))
        self.assertEqual(-7.4, maximum_of_lists(a, b, c))
        self.assertEqual(7.4, maximum_of_lists(a, b, c, abs_val=True))

    def test_normalize(self):
        a = torch.Tensor([-1.0, 1.0])
        b = torch.Tensor([-2.0, 1.5])

        self.assertEqual([1.0, -1.0], normalize(a)[0].tolist())
        self.assertEqual([-1.0, 1.0], normalize(a, abs_val=True)[0].tolist())
        self.assertEqual([0.5, -0.5], normalize(a, b)[0].tolist())
        self.assertEqual([-0.5, 0.5], normalize(a, b, abs_val=True)[0].tolist())

    def test_normalize_zeros(self):
        a = torch.Tensor([0.0, 0.0])
        b = torch.Tensor([0.0, 0.0])

        self.assertEqual([0.0, 0.0], normalize(a, b, abs_val=True)[0].tolist())

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

    def test_validate_reg_type(self):
        with self.assertRaises(AssertionError):
            validate_reg_type("abc")
        validate_reg_type("smoothgrad")
        validate_reg_type("vargrad")

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
