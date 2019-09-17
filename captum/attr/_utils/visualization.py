#!/usr/bin/env python3

from enum import Enum
import numpy as np


class VisualizeMethod(Enum):
    heat_map = 1
    masked_image = 2
    alpha_scaled = 3
    blended_heat_map = 4

class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

green = [0, 255, 0]
red = [255, 0, 0]
blue = [0, 0, 255]


def visualize_image_attr(attr, original_image=None, channel_transpose=False, sign="positive", method="heat_map",outlier_perc=2):
    if channel_transpose:
        attr = np.transpose(attr, (1,2,0))
        if original_image is not None:
            original_image = np.transpose(original_image, (1,2,0))

    # Combine RGB attribution channels
    attr_combined = np.sum(attr, axis=2)
    heat_map_color = None

    if VisualizeMethod[method] == VisualizeMethod.blended_heat_map:
        assert original_image is not None, "Image must be provided for blended heat map."
        return (0.6 * np.expand_dims(np.mean(original_image, axis=2),axis=2) + 0.4 * visualize_image_attr(attr=attr, original_image=original_image, channel_transpose=False, sign=sign, method="heat_map",outlier_perc=outlier_perc)).astype(int)

    if VisualizeSign[sign] == VisualizeSign.all:
        assert VisualizeMethod[method] == VisualizeMethod.heat_map, "Heat Map is the only supported visualization approach for both positive and negative attribution."
        return visualize_image_attr(attr=attr, original_image=original_image, channel_transpose=False, sign="positive", method=method,outlier_perc=outlier_perc) + visualize_image_attr(attr=attr, original_image=original_image, channel_transpose=False, sign="negative", method=method,outlier_perc=outlier_perc)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        attr_combined = attr_combined / np.percentile(attr_combined, 100 - outlier_perc)
        attr_combined[attr_combined > 1] = 1
        heat_map_color = green
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        attr_combined = attr_combined / np.percentile(attr_combined, outlier_perc)
        attr_combined[attr_combined > 1] = 1
        heat_map_color = red
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        attr_combined = attr_combined / np.percentile(attr_combined, 100 - outlier_perc)
        attr_combined[attr_combined > 1] = 1
        heat_map_color = blue
    else:
        raise AssertionError("Visualize Sign type is not valid.")

    # Apply chosen visualization method.
    if VisualizeMethod[method] == VisualizeMethod.heat_map:
        return (np.expand_dims(attr_combined, 2) * heat_map_color).astype(int)
    elif VisualizeMethod[method] == VisualizeMethod.masked_image:
        assert original_image is not None, "Image must be provided for masking."
        assert np.shape(original_image)[:-1] == np.shape(attr_combined), "Image dimensions do not match attribution dimensions for masking."
        return (np.expand_dims(attr_combined, 2) * original_image).astype(int)
    elif VisualizeMethod[method] == VisualizeMethod.alpha_scaled:
        assert original_image is not None, "Image must be provided for masking."
        assert np.shape(original_image)[:-1] == np.shape(attr_combined), "Image dimensions do not match attribution dimensions for adding alpha channel."
        # Concatenate alpha channel and return
        return np.concatenate((original_image, (255*np.expand_dims(attr_combined, 2)).astype(int)), axis=2)
    elif VisualizeMethod[method] == VisualizeMethod.blended_heat_map:
        return np.concatenate((original_image, (255*np.expand_dims(attr_combined, 2)).astype(int)), axis=2)
    else:
        raise AssertionError("Visualize Method type is not valid.")

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
