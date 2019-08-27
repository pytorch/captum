#!/usr/bin/env python3
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.widgets.helpers import DataEntry, ModelWrapper, wrap_dataset
from captum.widgets import ExploreWidget


class BasicModel_MultiTensor(nn.Module):
    def __init__(self):
        super(BasicModel_MultiTensor, self).__init__()

    def forward(self, input1, input2):
        input = torch.cat((input1, input2))
        return [1 - int(F.relu(1 - input)[0].item())]


class BasicWrap(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def get_prediction(self, input, meta=None):
        return self.model.forward(*input)

    def get_prediction_classes(self):
        return ["Class1", "Class2"]

    def get_feature_categories(self):
        return ["Image", "Default"]

    def get_feature_names(self):
        return ["G1", "G3"]

    def process_attributions(self, attributions):
        return attributions

    def get_prediction_indices(self, prediction):
        return [prediction]


test_model = BasicWrap(BasicModel_MultiTensor())
test_inputs = [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0])]
test_data = [
    DataEntry(["raw_inputs", "raw"], test_inputs, [0], None),
    DataEntry(["raw_inputs", "raw"], test_inputs, [1], None),
]


class Test(unittest.TestCase):
    def test_explore_loads(self):
        dataset = wrap_dataset(test_data, lambda x: x, test_model)
        print("test_data: ", dataset)
        e = ExploreWidget(test_model, [dataset])
        self.assertTrue("widgets" in str(type(e.final_view)))


if __name__ == "__main__":
    unittest.main()
