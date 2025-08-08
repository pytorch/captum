from captum.attr._core.rex import *

from captum.testing.helpers.basic import BaseTest
from parameterized import parameterized

import torch

class Test(BaseTest):
    # rename for convenience
    ts = torch.tensor

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
        rex = ReX(lambda x: True)

        attributions = rex.attribute(input, baseline, n_partitions=2)[0]
        if isinstance(input, tuple): input = input[0]
        print(attributions)
        # Forward_func returns a constant, no responsibility in input   
        self.assertFalse(torch.sum(attributions, dim=None))
        self.assertEqual(attributions.size(), input.size())


    @parameterized.expand([
        # input                                  # selected_idx
        (ts([1,2,3]),                            0),
        (ts([[1,2], [3,4]]),                     (0, 1)),
        (ts([[[1, 2], [3, 4]], [[5,6], [7,8]]]), (0, 1, 0))
    ])
    def test_selector_function(self, input, idx):
        rex = ReX(lambda x: x[idx])
        attributions = rex.attribute(input, 0)[0]
        print(attributions)
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
        print(attributions)
        self.assertTrue(attributions[idx])
        attributions[idx] = 0
        self.assertLess(torch.sum(attributions, dim=None), 1)

