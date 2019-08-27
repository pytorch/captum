#!/usr/bin/env python3
import unittest
import torch
import ipywidgets as widgets

from captum.widgets.visualization_helpers import (
    tab_constructor,
    search_constructor,
    popup_constructor,
    _class_selection,
    _multiple_item_selection,
    _label_hbox,
    _shim_html,
    words_constructor,
)
from captum.widgets.visualization import VisualizationEntry
from captum.widgets.helpers import DataEntry
from types import SimpleNamespace as sn

parent = sn(wrapped_model=sn(get_prediction_classes=lambda: ["dog", "cat", "animal"]))


class Test(unittest.TestCase):
    def test_visualization_entry_add(self):
        datum = DataEntry(
            (["The", "furrball", "was", "huge"],),
            (torch.tensor([1.0, 1.0, 1.0, 1.0])),
            [1, 2],
            None,
        )
        v = VisualizationEntry(datum, parent)
        v.add(1, True, True)
        v.add(2, True, False)
        v.add(0, False, False)
        self.assertEqual(3, len(v.rows))

    def test_widget_builders_tabs(self):
        w = tab_constructor(["tab1", "tab2"], [widgets.Box(), widgets.Button()])
        self.assertTrue("ipywidgets.widgets" in str(type(w)))

    def test_shim_html(self):
        _shim_html()

    def test_widget_builders_search(self):
        w = search_constructor(["class1", "class2", "class3"])
        self.assertTrue("ipywidgets.widgets" in str(type(w)))

    def test_widget_builders_popup(self):
        w = popup_constructor(widgets.Button(), widgets.HBox())
        self.assertTrue("ipywidgets.widgets" in str(type(w)))

    def test_widget_builders_class(self):
        w = _class_selection(["class1", "class2", "class3"])
        self.assertTrue("ipywidgets.widgets" in str(type(w)))

    def test_widget_builders_multiple(self):
        w = _multiple_item_selection(["class1", "class2", "class3"])
        self.assertTrue("ipywidgets.widgets" in str(type(w)))

    def test_widget_builders_label(self):
        w = _label_hbox("label_string", widgets.VBox())
        self.assertTrue("ipywidgets.widgets" in str(type(w)))

    def test_widget_builders_words(self):
        w = words_constructor(
            ["The", "Big", "Green", "Egg"], torch.tensor([0.5, 0.3, -0.2, 0.1])
        )
        self.assertTrue("ipywidgets.widgets" in str(type(w)))


if __name__ == "__main__":
    unittest.main()
