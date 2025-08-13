from captum.attr._core.rex import *

from captum.testing.helpers.basic import BaseTest
from parameterized import parameterized

import torch

class Test(BaseTest):
    # rename for convenience
    ts = torch.tensor
    
    depth_opts = range(4, 10)
    n_partition_opts = range(3, 5)
    n_search_opts = range(5, 15)
    is_contiguous_opts = [False, True]

    all_options = list(itertools.product(depth_opts, n_partition_opts, n_search_opts, is_contiguous_opts))

    @parameterized.expand([
            # inputs:                       baselines:
            (ts([1,2,3]),                   ts([[2,3], [3,4]])),
            ((ts([1]),ts([2]),ts([3])),     (ts([1]),ts([2]))),
            ((ts([1])),                     ()),
            ((),                            ts([1]))
    ])
    def test_input_baseline_mismatch_throws(self, input, baseline):
        rex = ReX(lambda x: 1/0) # dummy forward, should be unreachable
        with self.assertRaises(AssertionError):
            rex.attribute(input, baseline)


    @parameterized.expand([
        (ts([1,2,3]),                           0),
        (ts([[1,2,3], [4,5,6]]),                0),
        (ts([1,2,3,4]),                         ts([0,0,0,0])),
        (ts([[1, 2], [1,2]]),                   ts([[0,0], [0,0]])),
        (ts([[[1,2], [3,4]], [[5,6], [7,8]]]),  0),
        ((ts([1,2]), ts([3,4]), ts([5,6])),     (0, 0, 0)),
        ((ts([1,2]), ts([3,4]), ts([5,6])),     (ts([0,0]), ts([0,0]), ts([0,0]))),
        ((ts([1,2]), ts([3,4])),                (ts([0,0]), ts([0, 0]))),
    ])
    def test_valid_input_baseline(self, input, baseline):
        for o in self.all_options:
            rex = ReX(lambda x: True)

            attributions = rex.attribute(input, baseline, *o)[0]

            inp_unwrapped = input
            if isinstance(input, tuple): inp_unwrapped = input[0]

            # Forward_func returns a constant, no responsibility in input   
            self.assertFalse(torch.sum(attributions, dim=None))
            self.assertEqual(attributions.size(), inp_unwrapped.size())


    @parameterized.expand([
        # input                                  # selected_idx
        (ts([1,2,3]),                            0),
        (ts([[1,2], [3,4]]),                     (0, 1)),
        (ts([[[1, 2], [3, 4]], [[5,6], [7,8]]]), (0, 1, 0))
    ])
    def test_selector_function(self, input, idx):
        for o in self.all_options:
            rex = ReX(lambda x: x[idx])

            attributions = rex.attribute(input, 0, *o)[0]
            self.assertTrue(attributions[idx] == 1)

            attributions[idx] = 0
            self.assertFalse(torch.sum(attributions, dim=None))


    @parameterized.expand([
        # input shape                             # important idx
        ((4,4),                                   (0,0)),
        # ((12, 12, 12),                            (1,2,1)),
        # ((12, 12, 12, 6),                         (1,1,4,1)),
        ((1920, 1080),                            (1, 1)) # image-like
    ])
    def test_selector_function_large_input(self, input_shape, idx):
        rex = ReX(lambda x: x[idx])

        input = torch.ones(*input_shape)
        attributions = rex.attribute(input, 0, n_partitions=2, search_depth=10, n_searches=3)[0]
        self.assertTrue(attributions[idx])
        attributions[idx] = 0
        self.assertLess(torch.sum(attributions, dim=None), 1)

    @parameterized.expand([
        # input shape                           # lhs_idx   # rhs_idx
        ((2,4),                                (0,2),      (1,3))


    ])
    def test_boolean_or(self, input_shape, lhs_idx, rhs_idx):
        for o in self.all_options:
            rex = ReX(lambda x: max(x[lhs_idx], x[rhs_idx]))
            input = torch.ones(input_shape)
            
            attributions = rex.attribute(input, 0, *o)[0]

            self.assertTrue(attributions[lhs_idx] > 0.25, f"{attributions}")
            self.assertTrue(attributions[rhs_idx] > 0.25, f"{attributions}")

            attributions[lhs_idx] = 0
            attributions[rhs_idx] = 0
            self.assertTrue(torch.sum(attributions) < 1, f"{attributions}")
