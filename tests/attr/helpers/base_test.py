import unittest

import torch

class BaseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
