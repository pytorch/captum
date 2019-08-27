#!/usr/bin/env python3
import torch
import ipywidgets as widgets

from captum.attributions.integrated_gradients import IntegratedGradients
from captum.widgets.explore_ui import attr_class_selector
from captum.widgets.visualization_helpers import (
    stacked_bar_constructor,
    render_features,
)
from types import SimpleNamespace


class VisualizationEntry:
    """ A class for storing and rendering the individual visualizations that
        comprise an individual sample

        Inputs:
            data_entry: A widgets.helpers.DateEntry that represents the sample to
                be visualized
            parent_visualization: The parent container for the visualization (useful
                for passing data up to the parent)
    """

    def __init__(self, data_entry, parent_visualization):
        self.data_entry = data_entry
        self.rows = []
        self.parent = parent_visualization

        self._predicted = []
        self._targets = []
        self._attr_row_map = {}

        # TODO(tuck): add a check below to pass extra arguments to the model when
        #   meta data is specified
        def get_wrapped():
            return lambda *x: self.parent.wrapped_model.get_prediction(x)

        self.IG = IntegratedGradients(get_wrapped())

    def add(self, attr_index, is_target, is_prediction):
        """ Add an attribution record to the Visualization Entry (to be rendered)

        Inputs:
            attr_index (int): The index that was attributied to in the model
            is_target (bool): True if the sample was truly of the class at attr_index
            is_prediction (bool): True if the sample was predicted by the model
        """
        row = SimpleNamespace(
            index=attr_index,
            confidence=torch.tensor(0.0),
            attributions=[0.0 * input for input in self.data_entry.inputs],
            delta=torch.tensor(0.0),
        )
        self.rows.append(row)
        self._attr_row_map[attr_index] = len(self.rows) - 1

        if is_prediction:
            self._predicted.append(attr_index)
        if is_target:
            self._targets.append(attr_index)

    def _get_prediction_labels(self):
        """Return a list of the names for all rows marked true by is_predicted
            (self.add)"""
        name_array = self.parent.wrapped_model.get_prediction_classes()
        return [name_array[i] for i in self._predicted]

    def _get_target_labels(self):
        """Return a list of the names for all rows marked true by is_target
            (self.add)"""
        name_array = self.parent.wrapped_model.get_prediction_classes()
        return [name_array[i] for i in self._targets]

    def _get_header(self, link="www.facebook.com"):
        pred_prob = widgets.Label(value=f"Prediction Probability: 0.00")
        pred_prob.add_class("predprob")

        export_button = widgets.Button(description="Add to Export")
        export_button.add_class("hyperlink-style")
        export_button.add_class("med")
        export_button.add_class("margin-left-auto")

        def _export_callback(b):
            if b.description == "Add to Export":
                export_button.add_class("added")
                export_button.description = "Added to Export"
                self.parent.export_storage.append(self)
            else:
                export_button.remove_class("added")
                export_button.description = "Add to Export"
                id = self.parent.export_storage.index(self)
                del self.parent.export_storage[id]

        export_button.on_click(_export_callback)

        link_tag = widgets.Box()
        if link is not None:
            link_tag = widgets.HTML(
                value='<a class="hyperlink" target="blank" \
                href="https://{}">Source</a>'.format(
                    link
                )
            )

        return widgets.HBox([pred_prob, export_button, link_tag]), pred_prob

    def get_card(self):
        raw_feats = self.data_entry.raw_inputs
        types = self.parent.wrapped_model.get_feature_categories()
        names = self.parent.wrapped_model.get_feature_names()
        rendered_feats = render_features(raw_feats, None, types, names)

        card_header, pred_prob = self._get_header()
        card_header.add_class("card-header")
        card_base = widgets.HBox()
        card = widgets.VBox([card_header, card_base])
        card.add_class("card")

        def _attr_selected_callback(classname):
            id = self.parent.wrapped_model.get_prediction_classes().index(classname)
            row_index = self._attr_row_map[id]
            self._update_row(row_index)
            pred_prob.value = (
                f"Prediction Probability {self.rows[row_index].confidence:.2f}"
            )
            row = self.rows[row_index]
            rendered_feats = render_features(raw_feats, row.attributions, types, names)
            card_base.children = [
                class_selector,
                stacked_bar_constructor(
                    [
                        (name, attr[0].sum().item())
                        for name, attr in zip(names, row.attributions)
                    ]
                ),
                *rendered_feats,
            ]

        class_selector = attr_class_selector(
            self._get_prediction_labels(),
            self._get_target_labels(),
            _attr_selected_callback,
        )

        card_base.children = [class_selector, *rendered_feats]

        return card

    # TODO(tuck): create tests for this method going forward to ensure compatibility
    #   with the captum library

    def _get_attribution(self, target, options):
        """ Fetch the attributions for an item in the VisualizationEntry

            Inputs:
                target (int): The index in the output to attribute to
                options (dict): The settings to pass to Integrated Gradients
        """
        _, inputs, _, meta_data = self.data_entry

        try:
            self.parent.wrapped_model.model.zero_grad()
        except AttributeError:
            raise ValueError(
                "wrapped_model does not expose a pytorch model.\
                wrapped_model.model should be a pytorch model"
            )

        baseline_inputs = tuple(0.0 * input for input in inputs)

        start_confidence = self.parent.wrapped_model.get_prediction(
            inputs, meta=meta_data
        )[0][target]

        methods_defined = {"Riemann Sum": "riemann", "Gauss Legendre": "gausslegendre"}
        approx_method = methods_defined[options["approx_method"]]

        attributions, delta = self.IG.attribute(
            inputs,
            baselines=baseline_inputs,
            target=target,
            additional_forward_args=None,
            n_steps=options["approx_steps"],
            method=approx_method,
        )

        return SimpleNamespace(
            attributions=attributions, delta=delta, confidence=start_confidence
        )

    def _update_row(self, row_index):
        row = self.rows[row_index]
        if hasattr(row, "updated"):
            return
        else:
            attr_data = self._get_attribution(row.index, self.parent.options)
            row.attributions = attr_data.attributions
            row.confidence = attr_data.confidence
            row.delta = attr_data.delta
            row.updated = True


class Visualization:
    """ A class for storing multiple VisualizationEntry children and drawing them

        Inputs:
            wrapped_model (see captum.widgets.ModelWrapper): A wrapped version
                the model that is to be represented.
            options (dict): All options for changing the visualization. These are
                handled by the parent Widget.
    """

    def __init__(self, wrapped_model, export_storage, options=None):
        self.children = []
        self.wrapped_model = wrapped_model
        self.export_storage = export_storage
        self.options = options

    def new_child(self, data_entry):
        """ Creates a new child Visualization Entry to add to the renderer

            Inputs:
                data_entry (captum.Widgets.helpers.DataEntry): The example that
                    will be attributed, wrapped in a common interface.

            Returns the VisualizationEntry that was generated (for customization)
        """
        child = VisualizationEntry(data_entry, self)
        self.children.append(child)
        return child

    def get_cards(self):
        """Return a card view representation of the widget"""
        cards = []
        for child in self.children:
            cards.append(child.get_card())
        return cards

    def __len__(self):
        return len(self.children)

    def __getitem__(self, index):
        return self.children[index]
