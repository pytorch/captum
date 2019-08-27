#!/usr/bin/env python3

import ipywidgets as widgets

from captum.widgets.visualization_helpers import (
    stacked_bar_constructor,
    search_constructor,
    popup_constructor,
    tab_constructor,
    _shim_html,
    attr_class_selector,
    render_feature,
)


def get_header_view(parent, params):
    title = widgets.Label(value="Insights")
    title.add_class("title")

    model_label = widgets.Label(value="Model")
    model_label.add_class("modern-style")
    model_dropdown = widgets.Dropdown(options=["default"])
    model_dropdown.add_class("clean-dropdown")
    model_picker = widgets.HBox([model_label, model_dropdown])

    data_label = widgets.Label(value="Data")
    data_label.add_class("modern-style")
    dataset_dropdown = widgets.Dropdown(
        options=[dsn for dsn in params["dataset_names"]], op_id="dataset_name"
    )
    dataset_dropdown.add_class("clean-dropdown")
    dataset_picker = widgets.HBox([data_label, dataset_dropdown])

    ig_button = widgets.Button(description="Settings")
    ig_button.add_class("hyperlink-style")

    def ig_settings_click_callback(b):
        parent.state["settings_open"] = not parent.state["settings_open"]
        parent.update()

    ig_button.on_click(ig_settings_click_callback)
    ig_button.add_class("ig-button-right")

    header = widgets.HBox([title, model_picker, dataset_picker, ig_button])
    header.add_class("rel")
    return header


def get_settings_view(params):

    ig_approximization_function = widgets.Dropdown(
        options=["Riemann Sum", "Gauss Legendre"],
        value="Gauss Legendre",
        description="Aproximation Function:",
        op_id="approx_method",
    )
    ig_approximization_function.add_class("clean-dropdown")

    ig_approximization_step_count = widgets.IntText(
        value=10, description="Approximization Steps:", op_id="approx_steps"
    )
    ig_approximization_step_count.add_class("number-entry")

    sampling_method = widgets.Dropdown(
        options=["Random", "Consecutive"],
        value="Random",
        description="Sampling Method:",
        op_id="sampling_method",
    )
    sampling_method.add_class("clean-dropdown")

    ig_controls = widgets.VBox(
        [ig_approximization_function, ig_approximization_step_count]
    )

    sampling_controls = widgets.VBox([sampling_method])
    settings_wrapper = widgets.HBox([ig_controls, sampling_controls])
    settings_wrapper.add_class("settings-wrapper")
    return settings_wrapper


def get_instance_mock(predicted, actual, contributions, text):
    card = widgets.HBox()
    card.add_class("card")

    children = []
    children.append(attr_class_selector(predicted, actual, lambda: print("callback")))
    children.append(stacked_bar_constructor(contributions))
    text_attributions = [1.0, -0.2, 0.0, 0.0]
    children.append(
        render_feature(
            text,
            text_attributions,
            ftype="text-tokens",
            name="Text Feature",
            link="www.facebook.com",
        )
    )
    card.children = children
    return card


def get_instances(visualization):
    return visualization.get_cards()


def get_target_selector(params):
    targeted_button = widgets.Button(description="Filter by Class")
    targeted_button.add_class("hyperlink-style")
    targeted_button.add_class("small")

    heading1 = widgets.Label(value="Predicted")
    heading2 = widgets.Label(value="Targets")
    heading1.add_class("sub-heading")
    heading2.add_class("sub-heading")
    predictedsearch = search_constructor(
        params["classes"], placeholder="", op_id="clipping_ids_predicted"
    )
    targetssearch = search_constructor(
        params["classes"], placeholder="", op_id="clipping_ids_target"
    )

    target_options = widgets.VBox([heading1, predictedsearch, heading2, targetssearch])

    targetedpopup = popup_constructor(targeted_button, target_options)
    return targetedpopup


def get_instance_settings(params, fetch_storage):  # Where to store the fetched data
    targetedpopup = get_target_selector(params)

    instance_count = widgets.BoundedIntText(
        value=5, min=1, max=1000, op_id="num_records"
    )
    instance_count.add_class("number-entry")

    instance_selection = widgets.Dropdown(
        options=["All", "Correct", "Incorrect"], op_id="instance_type"
    )
    instance_selection.add_class("clean-dropdown")

    def _fetch_callback(b):
        vis = params["get_visualization"]()
        cards = get_instances(vis)
        fetch_storage.children = cards

    fetch_button = widgets.Button(description="Fetch")
    fetch_button.add_class("float-right")
    fetch_button.on_click(_fetch_callback)

    options_bar = widgets.HBox(
        [
            widgets.Label(value="Number of Instances"),
            instance_count,
            widgets.Label(value="Targeted Instances"),
            instance_selection,
            targetedpopup,
            fetch_button,
        ]
    )

    options_bar.children = [c.add_class("small-heading") for c in options_bar.children]
    options_bar.add_class("sub-settings")
    return options_bar


def get_instance_attr_view(params):
    instances_container = widgets.VBox()
    instances_container.add_class("card-back")

    options_bar = get_instance_settings(params, instances_container)

    view = widgets.VBox([options_bar, instances_container])
    view.add_class("wide")
    return view


def get_export_settings(params, fetch_storage):  # Where to store the fetched data
    targetedpopup = get_target_selector(params)

    instance_count = widgets.BoundedIntText(
        value=5, min=1, max=1000, op_id="num_records"
    )
    instance_count.add_class("number-entry")

    instance_selection = widgets.Dropdown(
        options=["All", "Correct", "Incorrect"], op_id="instance_type"
    )
    instance_selection.add_class("clean-dropdown")

    def _fetch_callback(b):
        vis = params["get_visualization"]()
        cards = get_instances(vis)
        fetch_storage.children = cards

    fetch_button = widgets.Button(description="Fetch")
    fetch_button.add_class("float-right")
    fetch_button.on_click(_fetch_callback)

    options_bar = widgets.HBox(
        [
            widgets.Label(value="Number of Instances"),
            instance_count,
            widgets.Label(value="Targeted Instances"),
            instance_selection,
            targetedpopup,
            fetch_button,
        ]
    )

    options_bar.children = [c.add_class("small-heading") for c in options_bar.children]
    options_bar.add_class("sub-settings")
    return options_bar


def get_export_view(params):
    instances_container = widgets.VBox([])
    instances_container.add_class("card-back")

    # options_bar = get_instance_settings(params, instances_container)

    view = widgets.VBox([instances_container])
    instances_container.children = [v.get_card() for v in params["export_container"]]
    view.add_class("wide")
    return view


def get_tab_views(params):
    export_view = widgets.Box([get_export_view(params)])

    def update_export():
        export_view.children = [get_export_view(params)]

    return tab_constructor(
        ["Instance Attribution ", "Export"],
        [get_instance_attr_view(params), export_view],
        callbacks=[None, update_export],
    )


def get_explore_widget(prediction_classes, parent_obj, params):
    Widget = widgets.VBox()
    Widget.state = {"settings_open": False}
    _shim_html()
    base = [get_header_view(Widget, params)]
    settings = get_settings_view({})
    tabs = get_tab_views(params)
    Widget.children = [*base, settings, tabs]

    def widget_update():
        if Widget.state["settings_open"]:
            Widget.children[1].remove_class("hidden")
        else:
            Widget.children[1].add_class("hidden")

    Widget.add_class("widget-tall")
    Widget.update = widget_update
    Widget.update()
    Widget.add_class("dumbledore")
    return Widget
