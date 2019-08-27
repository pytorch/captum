#!/usr/bin/env python3

import unittest

import ipywidgets as widgets
import torch

from captum.widgets.helpers import (
    DataEntry,
    ModelWrapper,
    Widget,
    EmptyDatasetException,
)
from captum.widgets.visualization_helpers import _shim_html, _label_hbox
from captum.widgets.visualization import Visualization


class BasicWrap(ModelWrapper):
    def __init__(self):
        self.model = None

    def get_prediction(self, input, meta_data):
        return torch.Tensor([1.0, 0.0])

    def get_prediction_classes(self):
        return ["Class1", "Class2"]

    def get_feature_categories(self):
        return ["Image", "Text", "Default"]

    def get_feature_names(self):
        return ["G1", "G2", "G3"]

    def process_attributions(self, attributions):
        return attributions


test_model = BasicWrap()


test_tensor = torch.tensor([1.0, 2.0])


test_data = [
    DataEntry(["raw image", "raw text", "raw"], [test_tensor] * 3, [0], None),
    DataEntry(["raw image", "raw text", "raw"], [test_tensor] * 3, [0, 1], None),
]


test_widget = Widget()


class Test(unittest.TestCase):
    def test_valid_model(self):
        test_widget._validate_input(test_model, [test_data])

    def test_invalid_dataset(self):
        with self.assertRaises(EmptyDatasetException):
            test_widget._validate_input(test_model, [])

    def test_visualization(self):
        v = Visualization(test_model, [])
        child = v.new_child(test_data[0])

        child.add(0, True, True)
        child.add(1, False, True)
        self.assertEqual(len(child.rows), 2)

        # child.get_widget()
        v.get_cards()

    def test_shim(self):
        _shim_html()

    def test_label_hbox(self):
        c = widgets.IntSlider()
        w = _label_hbox("Test", c)
        self.assertEqual(w.children[0].value, "Test")


if __name__ == "__main__":
    unittest.main()
