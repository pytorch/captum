#!/usr/bin/env python3
import matplotlib
import os
import ipywidgets as widgets
import torch

from IPython.core.display import display, HTML
from matplotlib import pyplot as plt


font = {"size": 15}
matplotlib.rc("font", **font)


class OffsetFeature:
    def __init__(self, names, offsets):
        self.names = tuple(names)
        self.offsets = tuple(offsets)
        self._name_to_index = {name: i for i, name in enumerate(names)}
        assert len(names) == len(offsets)

    def get_children_names(self):
        return self.names

    def get_child_tensor(self, t, child_name):
        i = self._name_to_index[child_name]
        try:
            return t[self.offsets[i] : self.offsets[i + 1]]
        except IndexError:
            return t[self.offsets[i] :]


def render_text_tokens(tokens, attributions):
    if attributions is None:
        attributions = torch.zeros(len(tokens))
        return words_constructor(tokens, attributions)
    else:
        return words_constructor(
            tokens, torch.tensor([t.sum().item() for t in attributions[0]])
        )


def render_split_tensor(offset_feature, attributions, name="Split"):
    if attributions is None:
        return widgets.Box()
    contributions = [
        (cn, offset_feature.get_child_tensor(attributions, cn).sum())
        for cn in offset_feature.get_children_names()
    ]
    return stacked_bar_constructor(contributions, name=name)


def render_default_feature(attributions):
    if attributions is not None:
        return widgets.Label(value=f"{attributions.sum().item():.2f}")
    return widgets.Label(value="0.00")


def render_feature(raw_feature, attributions, ftype=None, name="Feature"):
    heading = widgets.Label(value=name)
    heading.add_class("sub-heading")

    feature = widgets.Label(value="Not Yet Implemented")
    if ftype == "text_tokens":
        feature = render_text_tokens(raw_feature, attributions)

    if ftype == "offset_tensor":
        heading = widgets.Box()
        feature = render_split_tensor(raw_feature, attributions, name=name)

    if ftype == "plot_only":
        # TODO (tuck): Change to a return None and filter the returned list of
        # features
        return widgets.Box()

    if ftype == "default":
        feature = render_default_feature(attributions)

    return widgets.VBox([heading, feature])


def render_features(raw_features, attributions, types, names):
    if attributions is None:
        attributions = [None] * len(raw_features)
    return flow_features(
        [
            (render_feature(raw, attr, ftype=type, name=name), type)
            for raw, attr, type, name in zip(raw_features, attributions, types, names)
        ]
    )


def flow_features(feats):
    bar_items = []
    other = []
    for feature_view, type in feats:
        if type == "offest_tensor":
            bar_items.append(feature_view)
        else:
            other.append(feature_view)
    return [widgets.HBox(bar_items), widgets.VBox(other)]


def attr_class_selector(predicted, actual, callback):
    h1 = widgets.Label(value="Predicted")
    h2 = widgets.Label(value="Target")
    h1.add_class("sub-heading")
    h2.add_class("sub-heading")

    parent = widgets.HBox()

    s1 = _class_selection(predicted, grandparent=parent, callback=callback)
    s2 = _class_selection(actual, grandparent=parent, callback=callback)

    b1 = widgets.VBox([h1, s1])
    b2 = widgets.VBox([h2, s2])
    b1.add_class("attr-selector")
    b2.add_class("attr-selector")

    children = []
    if len(predicted) > 0:
        children.append(b1)
    if len(actual) > 0:
        children.append(b2)

    parent.children = children

    def clear_all():
        for child in children:
            child.children[1].clear()

    parent.clear_all = clear_all

    return parent


def _class_selection(categories, op_id=None, grandparent=None, callback=None):
    """Builder method for a class selection widget.

    inputs:
        categories (list of strings): The values to select from
        op_id (string): Key for storing value in options
        """
    buttons = []
    for category in categories:
        b = widgets.Button(description=category, tooltip="Click me")
        b.add_class("multi")
        buttons.append(b)
    parent = widgets.VBox(buttons)
    parent.value = None
    parent.op_id = op_id

    def _onclick(b):
        if grandparent:
            grandparent.clear_all()
        parent.value = b.description
        for button in buttons:
            if button is b:
                button.add_class("selected")
            else:
                button.remove_class("selected")
        if callback:
            d = b.description
            b.description = "Loading..."
            callback(d)
            b.description = d

    def clear():
        parent.value = None
        for button in buttons:
            button.remove_class("selected")

    parent.clear = clear

    for button in buttons:
        button.on_click(_onclick)
    return parent


def normalize(t, norm_method):
    modifier = 1.0
    if norm_method == "L1":
        modifier = sum(abs(t))
    elif norm_method == "L2":
        modifier = t.norm()
    elif norm_method == "Linf":
        modifier = abs(t).max()
    elif norm_method is not None:
        raise ValueError('Norm method "{}" is not defined'.format(norm_method))
    # be careful not to divide by 0
    if modifier < 0.01:  # TODO (tuck): should we ever scale a vector up?
        modifier = 1.0

    return t / float(modifier)


def piechart(categories, percentages):
    """Draws a piechart using matplotlib (as a side effect)

        Inputs:
            categories (list of str): The names of the slices on the
                chart.
            percentages (list of float): Representing the sizes
                of the slices in the chart. Should add up to 1.
    """
    categories = [
        "{} ({:.2f}%)".format(cat, 100 * pct)
        for cat, pct in zip(categories, percentages)
    ]
    plt.pie(
        percentages,
        labels=categories,
        colors=["#86f7f7", "#70ef69", "#fc9f48", "#e164f3", "#5b59ff"],
    )
    plt.axis("equal")
    plt.show()


def tab_constructor(names, child_views, callbacks=None):
    """Constructs tabs consisting of the child_views, named by names

    Inputs:
        names (list of str): The titles of the tabs
        child_views (ipywidgets.Widget): The views to render for each tab
        callbacks (list of func): Functions to call when switching to a given tab
            (None if tab doesn't need update function. List will be parsed positionally
            so that the first function goes with the first tab on click event)

    Returns a ipywidgets.Widget
    """
    assert len(names) == len(child_views)

    view_port = widgets.Box([child_views[0]])
    buttons = [widgets.Button(description=name) for name in names]
    buttons[0].add_class("targeted")
    tab_header = widgets.HBox(buttons)

    class Callback:
        def __init__(self, id):
            self.id = id

        def __call__(self, b):
            if callbacks:
                if callbacks[self.id]:
                    callbacks[self.id]()
            view_port.children = (child_views[self.id],)
            for button in buttons:
                if button is b:
                    button.add_class("targeted")
                else:
                    button.remove_class("targeted")

    for i, button in enumerate(buttons):
        button.add_class("tab-style")
        _click_callback = Callback(i)
        button.on_click(_click_callback)

    return widgets.VBox([tab_header, view_port])


def _tag_constructor(name, callback=None):
    """Constructs a tag (for placing inside of searchbar currently)

    Inputs:
        name (str): The label of the tag
        callback (func): The function to call when the tag's close button is
            pressed

    Returns:
        An ipywidget that contains the tag specified
    """
    label = widgets.Label(value=name)
    label.add_class("tag-label")

    close_button = widgets.Button(icon="times")
    close_button.add_class("close-button")
    close_button.parent_name = name

    container = widgets.HBox([label, close_button])
    container.add_class("tag-container")

    if callback is not None:
        close_button.on_click(callback)

    close_button.parent = container
    return container


def search_constructor(categories, placeholder="", op_id=None):
    """Construct a multi-tag search bar

    Inputs:
        categories (list of str): The categories to choose from.
        placholder (str): What to display in the placeholder of the search bar
        op_id (str): The title of the setting that will be used when building
            the options dict

    Returns a ipywidgets.Widget
    """
    categories = sorted(categories)

    searchbox = widgets.Text(placeholder=placeholder)
    searchbox.add_class("search-tool-box")

    tags = []
    searchbar = widgets.Box([searchbox])
    searchbar.add_class("search-tool-bar")

    resultsbox = widgets.VBox([])
    resultsbox.add_class("search-tool-results")
    resultsbox.add_class("hidden")

    search = widgets.VBox([searchbar, resultsbox])
    search.add_class("search-tool-container")
    if op_id is not None:
        search.op_id = op_id
    search.value = []

    def _results_button_callback(b):
        """Callback for whenever the user presses a result in the search results
            list. Adds the tag to the list of displayed tags and to the value of
            the widget"""
        text = b.description
        if text not in search.value:
            resultsbox.children = []
            searchbox.value = ""
            tag = _tag_constructor(
                b.description, callback=lambda b: _tag_close_callback(b)
            )
            tags.append(tag)
            searchbar.children = [*tags, searchbox]
            search.value.append(text)

    def _tag_close_callback(b):
        """Callback for whenever a tag is closed, removing from the tags displayed
            in the search bar, and from the value of the widget"""
        text = b.parent_name
        search.value.remove(text)
        tags.remove(b.parent)
        searchbar.children = [*tags, searchbox]

    def _update_search(w):
        """Callback for updating results based on the text that a user has currently
            typed in. Fetches all valid results and displays them."""
        text = w.new
        if text.strip() == "":
            resultsbox.children = []
            resultsbox.add_class("hidden")
            return
        resultsbox.remove_class("hidden")
        results = [
            widgets.Button(description=cat)
            for cat in categories
            if cat.lower().startswith(text.lower())
        ]
        for b in results:
            b.on_click(_results_button_callback)
        resultsbox.children = results

    searchbox.observe(_update_search, "value")

    return search


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def words_constructor(words, attributions):
    """Constructs a box containing attributions rendered over text
            (and popups containing values of attributions)

        Inputs:
            words (list of str): The words to render
            attributions (list of float): The attribution scores of the words
                to render

        Returns an Ipywidgets.widget containing the requested view
    """
    word_chunks = []
    attributions_norm = normalize(attributions, "L2")
    pos_color = _get_color(1.0)
    neg_color = _get_color(-1.0)
    for word, attr, attrnorm in zip(words, attributions, attributions_norm):
        color = _get_color(attrnorm)
        html = """
        <div class="word-box">
            <div class="word" style="background: {color};">
                <div class="word-box-popup">
                    <p class="popup-label">Attribution Score</p>
                    <p class="popup-attr" style="color: {numcolor};">{attr:.2f}</p>
                </div>
                {word}
            </div>
        </div>
        """.format(
            color=color,
            word=word,
            attr=attr,
            numcolor=pos_color if attrnorm >= 0 else neg_color,
            attrnorm=attrnorm,
        )
        word_chunks.append(html)
    w = widgets.HTML(value="".join(word_chunks))
    w.add_class("word-box-container")
    return w


def popup_constructor(clickable, popup, label_updater=None):
    """Construct a popup from a button and a popup view. When the clickable is
        clicked, the popup view will render below it. It will disapear when
        clicked again.

    Inputs:
        clickable (ipywidgets.Widget): Any widget that can receive an on_click
            method.
        popup (ipywidgets.Widget): The view to show on click. Will be wrapped
            in a container.
        label_updater (func): A function that updates the label. Will be called
            on close and open. (useful for adding counters to the label)

    Returns a ipywidgets.Widget
    """
    popup.add_class("hidden")
    popup.add_class("popup-under")
    container = widgets.Box([clickable, popup])
    container.add_class("popup-parent")

    clickable.popup_is_hidden = True

    def popup_callback(b):
        if label_updater:
            label_updater(b, popup.value)
        b.popup_is_hidden = not b.popup_is_hidden
        if b.popup_is_hidden:
            popup.add_class("hidden")
        else:
            popup.remove_class("hidden")

    clickable.on_click(popup_callback)
    return container


def _shim_html():
    """Fix to allow custom widget styling

    Ipywidgets doesn't allow custom styling for widgets, but this shim
        solves the issue, adding the stylesheet found at widget_stylesheet.css
        to the notebook. Added styling should be used sparingly, since it may
        break the widget.
    """
    style_path = os.path.join(os.path.dirname(__file__), "widget_stylesheet.css")
    with open(style_path) as style_file:
        stylesheet = style_file.read()
    html_shim = "<style>{}</style>".format(stylesheet)
    display(HTML(html_shim))


def _multiple_item_selection(choices, op_id=None):
    """Builder method for a search and multiselect widget.

    Takes in choices, and returns a widget capable of searching through those
    choices and adding them to a selection list"""
    search = widgets.Text(value="", placeholder="Search", description="")

    base = widgets.SelectMultiple(options=choices, value=[], description="")

    def search_func(b):
        if search.value == "":
            base.options = choices
        else:
            base.options = [
                opt for opt in choices if opt.lower().startswith(search.value.lower())
            ]

    search.on_submit(search_func)

    add = widgets.Button(description="->", tooltip="Add to list")

    def add_callback(b):
        items_to_add = base.value
        selected.options += items_to_add
        selected.options = list(dict.fromkeys(selected.options))

    add.on_click(add_callback)

    remove = widgets.Button(description="<-", tooltip="Remove from list")

    def remove_callback(b):
        items_to_remove = selected.value
        selected.options = [
            opt for opt in selected.options if opt not in items_to_remove
        ]

    remove.on_click(remove_callback)

    selected = widgets.SelectMultiple(
        options=[], value=[], description="", op_id_values=op_id
    )

    return widgets.HBox(
        [
            widgets.VBox([search, base]),
            widgets.VBox([add, remove]),
            widgets.VBox([selected]),
        ]
    )


def _label_hbox(label_string, widget):
    """Method that applies full width labels to a widget, returning a HBox with
        the label and widget inside"""
    label = widgets.Label(value=label_string, layout={"width": "100%"})
    return widgets.HBox([label, widget])


def _get_all_contributions(contributions):
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    children = []
    for name, cont in contributions:
        children.append(widgets.Label(value=f"{name}: {cont:.2f}"))
    view = widgets.VBox(children)
    view.add_class("expanded-cont")
    return view


def _get_bars(contributions, total_height, cut_off=0.01, min_height=10):
    significant = [c for c in contributions if c[1] < -cut_off or c[1] > cut_off]
    sig_num = len(significant)
    height_budget = total_height - min_height * sig_num

    if height_budget < 0:
        total_height = min_height * sig_num
        height_budget = 0

    abs_mag = sum(abs(v) for (name, v) in significant)
    return [
        (name, (v / abs_mag), height_budget * abs(v / abs_mag) + min_height, v)
        for (name, v) in significant
    ]


def stacked_bar_constructor(contributions, total_height=130, name="Contributions"):
    """
    Inputs:
        contributions (tuple list): A list of tuples of the form (name (string),
            contribution (float)) where contribution will decide the size and color
            of the bar [can be any value as long as magnitude is conserved].
        total_height: height of the chart in pixels.
    """
    heading = widgets.Label(value=name)
    heading.add_class("sub-heading")

    expand = widgets.Button(icon="question-circle")
    expand.add_class("small-button")
    full_cont = popup_constructor(expand, _get_all_contributions(contributions))

    contributions.sort(key=lambda x: x[1], reverse=True)

    children = []
    for name, pct, height, attr in _get_bars(contributions, total_height):
        hue = 0 if pct < 0 else 110
        bar = widgets.HTML(
            """
        <div class="bar-piece" style="background: hsl({hue}, 100%, {lightness}%);
            height:{height}px">
        </div>
        """.format(
                hue=hue, lightness=90 - abs(pct) * 40, height=height
            )
        )

        label = widgets.Label(
            value="{} {:.0f}% ({:.2f})".format(name, 100 * abs(pct), attr)
        )
        label.add_class("bar-label")
        children.append(widgets.HBox([bar, label]))

    chart = widgets.VBox([widgets.HBox([heading, full_cont]), *children])
    chart.add_class("bar-stack")
    return chart
