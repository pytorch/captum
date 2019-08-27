#!/usr/bin/env python3
from captum.widgets.helpers import Widget
from captum.widgets.visualization import Visualization
from captum.widgets.explore_ui import get_explore_widget
from IPython.core.display import display


class ExploreWidget(Widget):
    """IPython widget for interactive tablization of attribution data

    Args:
        model:
            A pytorch model wrapped in a ModelWrapper
            [instance of captum.Widgets.helpers.ModelWrapper]

        datasets:
            A list of wrapped datasets, (defined in helpers.py).
            User can call wrap_dataset to generate one.
    """

    def __init__(self, wrapped_model, datasets):
        self._validate_input(wrapped_model, datasets)

        self.wrapped_model = wrapped_model
        self.datasets = datasets
        self.dataset_names = [ds.name for ds in datasets]
        self.export = []

        params = {
            "binary_model": False,
            "classes": self.wrapped_model.get_prediction_classes(),
            "models": [wrapped_model],
            "dataset_names": self.dataset_names,
            "get_visualization": self._get_data,
            "instance_fetcher": None,
            "export_container": self.export,
        }

        self.final_view = get_explore_widget(
            wrapped_model.get_prediction_classes(), self, params
        )

    def get_root(self):
        """Return the root node of the widget, for compositional purposes"""
        return self.final_view

    def render(self):
        """Display the widget using the IPython display system"""
        display(self.final_view)

    def _get_data(self, loading=None):
        """Fetch data from captum for visualization.

        Inputs:
            loading (widget): A loading bar to update as the function progresses.

        Returns: a Visualization object
        """
        options = _get_options(self.final_view)
        visualization = Visualization(self.wrapped_model, self.export, options=options)

        ds_name = options["dataset_name"]
        dsid = self.dataset_names.index(ds_name)
        dataset = self.datasets[dsid]

        target_clipping_labels = options["clipping_ids_target"]
        predicted_clipping_labels = options["clipping_ids_predicted"]
        is_random = True if options["sampling_method"] == "Random" else False
        for index in dataset.find(
            target_clipping_labels,
            predicted_clipping_labels,
            random=is_random,
            limit=options["num_records"],
            exact=False,
            instance_type=options["instance_type"],
        ):
            data_entry = dataset.items[index]
            (raw_inputs, inputs, target_indices, metadata) = data_entry

            predicted_indices = dataset._predicted_indices[index]

            attr_indices = {*target_indices, *predicted_indices}
            entry = visualization.new_child(data_entry)
            for attr_index in attr_indices:
                entry.add(
                    attr_index,
                    attr_index in target_indices,
                    attr_index in predicted_indices,
                )
            if loading:
                loading.value = len(visualization) / float(options["num_records"])

        return visualization


def _get_options(root):
    """Combes through a widget tree and pulls out values given the following
    logic.

    Widget has attr 'op_id': store widget.value in options[op_id]

    Widget has attr 'op_id_values': store widget.options in options[op_id]
    """
    queue = [root]
    options = {}
    while queue:
        current_widget = queue.pop()
        if hasattr(current_widget, "op_id"):
            op_id = current_widget.op_id
            options[op_id] = current_widget.value

        if hasattr(current_widget, "op_id_values"):
            op_id = current_widget.op_id_values
            options[op_id] = current_widget.options

        if hasattr(current_widget, "children"):
            for child in current_widget.children:
                queue.append(child)
    return options
