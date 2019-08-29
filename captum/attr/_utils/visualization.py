#!/usr/bin/env python3

# TODO: move it to widget folder?

# First visualizations are based on MIT-licensed code from
# https://github.com/TianhongDai/integrated-gradient-pytorch
# This file copies some of the content from:
# https://github.com/TianhongDai/integrated-gradient-pytorch/
# blob/master/visualization.py

import numpy as np
from IPython.core.display import display, HTML


G = [0, 255, 0]
R = [255, 0, 0]


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


def linear_transform(
    attributions,
    clip_above_percentile=99.9,
    clip_below_percentile=70.0,
    low=0.2,
    plot_distribution=False,
):
    m = compute_threshold_by_top_percentage(
        attributions,
        percentage=100 - clip_above_percentile,
        plot_distribution=plot_distribution,
    )
    e = compute_threshold_by_top_percentage(
        attributions,
        percentage=100 - clip_below_percentile,
        plot_distribution=plot_distribution,
    )
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= transformed >= low
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def compute_threshold_by_top_percentage(
    attributions, percentage=60, plot_distribution=True
):
    if percentage < 0 or percentage > 100:
        raise ValueError("percentage must be in [0, 100]")
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError
    return threshold


def polarity_function(attributions, polarity):
    if polarity == "positive":
        return np.clip(attributions, 0, 1)
    elif polarity == "negative":
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def visualize_image(
    attributions,
    image,
    positive_channel=G,
    negative_channel=R,
    polarity="positive",
    clip_above_percentile=99.9,
    clip_below_percentile=0,
    morphological_cleanup=False,
    outlines=False,
    outlines_component_percentage=90,
    overlay=True,
    mask_mode=False,
    plot_distribution=False,
):
    if polarity == "both":
        raise NotImplementedError

    elif polarity == "positive":
        attributions = polarity_function(attributions, polarity=polarity)
        channel = positive_channel

    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(
        attributions,
        clip_above_percentile,
        clip_below_percentile,
        0.0,
        plot_distribution=plot_distribution,
    )
    attributions_mask = attributions.copy()
    if morphological_cleanup:
        raise NotImplementedError
    if outlines:
        raise NotImplementedError
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        if not mask_mode:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 2)
            attributions = np.clip(attributions * image, 0, 255)
            attributions = attributions[:, :, (2, 1, 0)]
    return attributions


# These visualization methods are for text and are partially copied from
# experiments conducted by Davide Testuggine at Facebook.


class VisualizationDataRecord:
    r"""
        A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "pred_prob",
        "pred_class",
        "target_class",
        "attr_class",
        "attr_score",
        "raw_input",
        "convergence_score",
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        target_class,
        attr_class,
        attr_score,
        raw_input,
        convergence_score,
    ):
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.target_class = target_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input = raw_input
        self.convergence_score = convergence_score


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


def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_tooltip(item, text):
    return '<div class="tooltip">{item}\
        <span class="tooltiptext">{text}</span>\
        </div>'.format(
        item=item, text=text
    )


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def visualize_text(datarecords: VisualizationDataRecord):
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Target Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(datarecord.target_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")
    display(HTML("".join(dom)))
