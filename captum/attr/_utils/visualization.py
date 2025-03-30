#!/usr/bin/env python3
import warnings
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray

try:
    from IPython.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class ImageVisualizationMethod(Enum):
    heat_map = 1
    blended_heat_map = 2
    original_image = 3
    masked_image = 4
    alpha_scaling = 5


class TimeseriesVisualizationMethod(Enum):
    overlay_individual = 1
    overlay_combined = 2
    colored_graph = 3


class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def _prepare_image(attr_visual: ndarray):
    return np.clip(attr_visual.astype(int), 0, 255)


def _normalize_scale(attr: ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_attr(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)


def visualize_image_attr(
    attr: ndarray,
    original_image: Union[None, ndarray] = None,
    method: str = "heat_map",
    sign: str = "absolute_value",
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
):
    r"""
    Visualizes attribution for a given image by normalizing attribution values
    of the desired sign (positive, negative, absolute value, or all) and displaying
    them using the desired mode in a matplotlib figure.

    Args:

        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (H, W, C), with
                    channels as last dimension. Shape must also match that of
                    the original image if provided.
        original_image (numpy.ndarray, optional): Numpy array corresponding to
                    original image. Shape must be in the form (H, W, C), with
                    channels as the last dimension. Image can be provided either
                    with float values in range 0-1 or int values between 0-255.
                    This is a necessary argument for any visualization method
                    which utilizes the original image.
                    Default: None
        method (str, optional): Chosen method for visualizing attribution.
                    Supported options are:

                    1. `heat_map` - Display heat map of chosen attributions

                    2. `blended_heat_map` - Overlay heat map over greyscale
                       version of original image. Parameter alpha_overlay
                       corresponds to alpha of heat map.

                    3. `original_image` - Only display original image.

                    4. `masked_image` - Mask image (pixel-wise multiply)
                       by normalized attribution values.

                    5. `alpha_scaling` - Sets alpha channel of each pixel
                       to be equal to normalized attribution value.

                    Default: `heat_map`
        sign (str, optional): Chosen sign of attributions to visualize. Supported
                    options are:

                    1. `positive` - Displays only positive pixel attributions.

                    2. `absolute_value` - Displays absolute value of
                       attributions.

                    3. `negative` - Displays only negative pixel attributions.

                    4. `all` - Displays both positive and negative attribution
                       values. This is not supported for `masked_image` or
                       `alpha_scaling` modes, since signed information cannot
                       be represented in these modes.

                    Default: `absolute_value`
        plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                    on which to visualize. If None is provided, then a new figure
                    and axis are created.
                    Default: None
        outlier_perc (float or int, optional): Top attribution values which
                    correspond to a total of outlier_perc percentage of the
                    total attribution are set to 1 and scaling is performed
                    using the minimum of these values. For sign=`all`, outliers
                    and scale value are computed using absolute value of
                    attributions.
                    Default: 2
        cmap (str, optional): String corresponding to desired colormap for
                    heatmap visualization. This defaults to "Reds" for negative
                    sign, "Blues" for absolute value, "Greens" for positive sign,
                    and a spectrum from red to green for all. Note that this
                    argument is only used for visualizations displaying heatmaps.
                    Default: None
        alpha_overlay (float, optional): Alpha to set for heatmap when using
                    `blended_heat_map` visualization mode, which overlays the
                    heat map over the greyscaled original image.
                    Default: 0.5
        show_colorbar (bool, optional): Displays colorbar for heatmap below
                    the visualization. If given method does not use a heatmap,
                    then a colormap axis is created and hidden. This is
                    necessary for appropriate alignment when visualizing
                    multiple plots, some with colorbars and some without.
                    Default: False
        title (str, optional): Title string for plot. If None, no title is
                    set.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (6,6)
        use_pyplot (bool, optional): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.

    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.

    Examples::

        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> ig = IntegratedGradients(net)
        >>> # Computes integrated gradients for class 3 for a given image .
        >>> attribution, delta = ig.attribute(orig_image, target=3)
        >>> # Displays blended heat map visualization of computed attributions.
        >>> _ = visualize_image_attr(attribution, orig_image, "blended_heat_map")
    """
    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = _prepare_image(original_image * 255)
    elif ImageVisualizationMethod[method] != ImageVisualizationMethod.heat_map:
        raise ValueError(
            "Original Image must be provided for"
            "any visualization other than heatmap."
        )

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    plt_axis.grid(b=False)

    heat_map = None
    # Show original image
    if ImageVisualizationMethod[method] == ImageVisualizationMethod.original_image:
        assert (
            original_image is not None
        ), "Original image expected for original_image method."
        if len(original_image.shape) > 2 and original_image.shape[2] == 1:
            original_image = np.squeeze(original_image, axis=2)
        plt_axis.imshow(original_image)
    else:
        # Choose appropriate signed attributions and normalize.
        norm_attr = _normalize_attr(attr, sign, outlier_perc, reduction_axis=2)

        # Set default colormap and bounds based on sign.
        if VisualizeSign[sign] == VisualizeSign.all:
            default_cmap = LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
        elif VisualizeSign[sign] == VisualizeSign.positive:
            default_cmap = "Greens"
            vmin, vmax = 0, 1
        elif VisualizeSign[sign] == VisualizeSign.negative:
            default_cmap = "Reds"
            vmin, vmax = 0, 1
        elif VisualizeSign[sign] == VisualizeSign.absolute_value:
            default_cmap = "Blues"
            vmin, vmax = 0, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")
        cmap = cmap if cmap is not None else default_cmap

        # Show appropriate image visualization.
        if ImageVisualizationMethod[method] == ImageVisualizationMethod.heat_map:
            heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
        elif (
            ImageVisualizationMethod[method]
            == ImageVisualizationMethod.blended_heat_map
        ):
            assert (
                original_image is not None
            ), "Original Image expected for blended_heat_map method."
            plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.masked_image:
            assert VisualizeSign[sign] != VisualizeSign.all, (
                "Cannot display masked image with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                _prepare_image(original_image * np.expand_dims(norm_attr, 2))
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.alpha_scaling:
            assert VisualizeSign[sign] != VisualizeSign.all, (
                "Cannot display alpha scaling with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                np.concatenate(
                    [
                        original_image,
                        _prepare_image(np.expand_dims(norm_attr, 2) * 255),
                    ],
                    axis=2,
                )
            )
        else:
            raise AssertionError("Visualize Method type is not valid.")

    # Add colorbar. If given method is not a heatmap and no colormap is relevant,
    # then a colormap axis is created and hidden. This is necessary for appropriate
    # alignment when visualizing multiple plots, some with heatmaps and some
    # without.
    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")
    if title:
        plt_axis.set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


def visualize_image_attr_multiple(
    attr: ndarray,
    original_image: Union[None, ndarray],
    methods: List[str],
    signs: List[str],
    titles: Union[None, List[str]] = None,
    fig_size: Tuple[int, int] = (8, 6),
    use_pyplot: bool = True,
    **kwargs: Any,
):
    r"""
    Visualizes attribution using multiple visualization methods displayed
    in a 1 x k grid, where k is the number of desired visualizations.

    Args:

        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (H, W, C), with
                    channels as last dimension. Shape must also match that of
                    the original image if provided.
        original_image (numpy.ndarray, optional): Numpy array corresponding to
                    original image. Shape must be in the form (H, W, C), with
                    channels as the last dimension. Image can be provided either
                    with values in range 0-1 or 0-255. This is a necessary
                    argument for any visualization method which utilizes
                    the original image.
        methods (list[str]): List of strings of length k, defining method
                        for each visualization. Each method must be a valid
                        string argument for method to visualize_image_attr.
        signs (list[str]): List of strings of length k, defining signs for
                        each visualization. Each sign must be a valid
                        string argument for sign to visualize_image_attr.
        titles (list[str], optional): List of strings of length k, providing
                    a title string for each plot. If None is provided, no titles
                    are added to subplots.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (8, 6)
        use_pyplot (bool, optional): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.
        **kwargs (Any, optional): Any additional arguments which will be passed
                    to every individual visualization. Such arguments include
                    `show_colorbar`, `alpha_overlay`, `cmap`, etc.


    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.

    Examples::

        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> ig = IntegratedGradients(net)
        >>> # Computes integrated gradients for class 3 for a given image .
        >>> attribution, delta = ig.attribute(orig_image, target=3)
        >>> # Displays original image and heat map visualization of
        >>> # computed attributions side by side.
        >>> _ = visualize_image_attr_multiple(attribution, orig_image,
        >>>                     ["original_image", "heat_map"], ["all", "positive"])
    """
    assert len(methods) == len(signs), "Methods and signs array lengths must match."
    if titles is not None:
        assert len(methods) == len(titles), (
            "If titles list is given, length must " "match that of methods list."
        )
    if use_pyplot:
        plt_fig = plt.figure(figsize=fig_size)
    else:
        plt_fig = Figure(figsize=fig_size)
    plt_axis = plt_fig.subplots(1, len(methods))

    # When visualizing one
    if len(methods) == 1:
        plt_axis = [plt_axis]

    for i in range(len(methods)):
        visualize_image_attr(
            attr,
            original_image=original_image,
            method=methods[i],
            sign=signs[i],
            plt_fig_axis=(plt_fig, plt_axis[i]),
            use_pyplot=False,
            title=titles[i] if titles else None,
            **kwargs,
        )
    plt_fig.tight_layout()
    if use_pyplot:
        plt.show()
    return plt_fig, plt_axis


def visualize_timeseries_attr(
    attr: ndarray,
    data: ndarray,
    x_values: Optional[ndarray] = None,
    method: str = "individual_channels",
    sign: str = "absolute_value",
    channel_labels: Optional[List[str]] = None,
    channels_last: bool = True,
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.7,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
    **pyplot_kwargs,
):
    r"""
    Visualizes attribution for a given timeseries data by normalizing
    attribution values of the desired sign (positive, negative, absolute value,
    or all) and displaying them using the desired mode in a matplotlib figure.

    Args:

        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (N, C) with channels
                    as last dimension, unless `channels_last` is set to True.
                    Shape must also match that of the timeseries data.
        data (numpy.ndarray): Numpy array corresponding to the original,
                    equidistant timeseries data. Shape must be in the form
                    (N, C) with channels as last dimension, unless
                    `channels_last` is set to true.
        x_values (numpy.ndarray, optional): Numpy array corresponding to the
                    points on the x-axis. Shape must be in the form (N, ). If
                    not provided, integers from 0 to N-1 are used.
                    Default: None
        method (str, optional): Chosen method for visualizing attributions
                    overlaid onto data. Supported options are:

                    1. `overlay_individual` - Plot each channel individually in
                        a separate panel, and overlay the attributions for each
                        channel as a heat map. The `alpha_overlay` parameter
                        controls the alpha of the heat map.

                    2. `overlay_combined` - Plot all channels in the same panel,
                        and overlay the average attributions as a heat map.

                    3. `colored_graph` - Plot each channel in a separate panel,
                        and color the graphs according to the attribution
                        values. Works best with color maps that does not contain
                        white or very bright colors.

                    Default: `overlay_individual`
        sign (str, optional): Chosen sign of attributions to visualize.
                    Supported options are:

                    1. `positive` - Displays only positive pixel attributions.

                    2. `absolute_value` - Displays absolute value of
                        attributions.

                    3. `negative` - Displays only negative pixel attributions.

                    4. `all` - Displays both positive and negative attribution
                        values.

                    Default: `absolute_value`
        channel_labels (list[str], optional): List of labels
                    corresponding to each channel in data.
                    Default: None
        channels_last (bool, optional): If True, data is expected to have
                    channels as the last dimension, i.e. (N, C). If False, data
                    is expected to have channels first, i.e. (C, N).
                    Default: True
        plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                    on which to visualize. If None is provided, then a new figure
                    and axis are created.
                    Default: None
        outlier_perc (float or int, optional): Top attribution values which
                    correspond to a total of outlier_perc percentage of the
                    total attribution are set to 1 and scaling is performed
                    using the minimum of these values. For sign=`all`, outliers
                    and scale value are computed using absolute value of
                    attributions.
                    Default: 2
        cmap (str, optional): String corresponding to desired colormap for
                    heatmap visualization. This defaults to "Reds" for negative
                    sign, "Blues" for absolute value, "Greens" for positive sign,
                    and a spectrum from red to green for all. Note that this
                    argument is only used for visualizations displaying heatmaps.
                    Default: None
        alpha_overlay (float, optional): Alpha to set for heatmap when using
                    `blended_heat_map` visualization mode, which overlays the
                    heat map over the greyscaled original image.
                    Default: 0.7
        show_colorbar (bool): Displays colorbar for heat map below
                    the visualization.
        title (str, optional): Title string for plot. If None, no title is
                    set.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (6,6)
        use_pyplot (bool): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.
        pyplot_kwargs: Keyword arguments forwarded to plt.plot, for example
                    `linewidth=3`, `color='black'`, etc

    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.

    Examples::

        >>> # Classifier takes input of shape (batch, length, channels)
        >>> model = Classifier()
        >>> dl = DeepLift(model)
        >>> attribution = dl.attribute(data, target=0)
        >>> # Pick the first sample and plot each channel in data in a separate
        >>> # panel, with attributions overlaid
        >>> visualize_timeseries_attr(attribution[0], data[0], "overlay_individual")
    """

    # Check input dimensions
    assert len(attr.shape) == 2, "Expected attr of shape (N, C), got {}".format(
        attr.shape
    )
    assert len(data.shape) == 2, "Expected data of shape (N, C), got {}".format(
        attr.shape
    )

    # Convert to channels-first
    if channels_last:
        attr = np.transpose(attr)
        data = np.transpose(data)

    num_channels = attr.shape[0]
    timeseries_length = attr.shape[1]

    if num_channels > timeseries_length:
        warnings.warn(
            "Number of channels ({}) greater than time series length ({}), "
            "please verify input format".format(num_channels, timeseries_length)
        )

    num_subplots = num_channels
    if (
        TimeseriesVisualizationMethod[method]
        == TimeseriesVisualizationMethod.overlay_combined
    ):
        num_subplots = 1
        attr = np.sum(attr, axis=0)  # Merge attributions across channels

    if x_values is not None:
        assert (
            x_values.shape[0] == timeseries_length
        ), "x_values must have same length as data"
    else:
        x_values = np.arange(timeseries_length)

    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(
                figsize=fig_size, nrows=num_subplots, sharex=True
            )
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots(nrows=num_subplots, sharex=True)

    if not isinstance(plt_axis, ndarray):
        plt_axis = np.array([plt_axis])

    norm_attr = _normalize_attr(attr, sign, outlier_perc, reduction_axis=None)

    # Set default colormap and bounds based on sign.
    if VisualizeSign[sign] == VisualizeSign.all:
        default_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )
        vmin, vmax = -1, 1
    elif VisualizeSign[sign] == VisualizeSign.positive:
        default_cmap = "Greens"
        vmin, vmax = 0, 1
    elif VisualizeSign[sign] == VisualizeSign.negative:
        default_cmap = "Reds"
        vmin, vmax = 0, 1
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        default_cmap = "Blues"
        vmin, vmax = 0, 1
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    cmap = cmap if cmap is not None else default_cmap
    cmap = cm.get_cmap(cmap)
    cm_norm = colors.Normalize(vmin, vmax)

    def _plot_attrs_as_axvspan(attr_vals, x_vals, ax):

        half_col_width = (x_values[1] - x_values[0]) / 2.0
        for icol, col_center in enumerate(x_vals):
            left = col_center - half_col_width
            right = col_center + half_col_width
            ax.axvspan(
                xmin=left,
                xmax=right,
                facecolor=(cmap(cm_norm(attr_vals[icol]))),
                edgecolor=None,
                alpha=alpha_overlay,
            )

    if (
        TimeseriesVisualizationMethod[method]
        == TimeseriesVisualizationMethod.overlay_individual
    ):

        for chan in range(num_channels):

            plt_axis[chan].plot(x_values, data[chan, :], **pyplot_kwargs)
            if channel_labels is not None:
                plt_axis[chan].set_ylabel(channel_labels[chan])

            _plot_attrs_as_axvspan(norm_attr[chan], x_values, plt_axis[chan])

        plt.subplots_adjust(hspace=0)

    elif (
        TimeseriesVisualizationMethod[method]
        == TimeseriesVisualizationMethod.overlay_combined
    ):

        # Dark colors are better in this case
        cycler = plt.cycler("color", cm.Dark2.colors)
        plt_axis[0].set_prop_cycle(cycler)

        for chan in range(num_channels):
            label = channel_labels[chan] if channel_labels else None
            plt_axis[0].plot(x_values, data[chan, :], label=label, **pyplot_kwargs)

        _plot_attrs_as_axvspan(norm_attr, x_values, plt_axis[0])

        plt_axis[0].legend(loc="best")

    elif (
        TimeseriesVisualizationMethod[method]
        == TimeseriesVisualizationMethod.colored_graph
    ):

        for chan in range(num_channels):

            points = np.array([x_values, data[chan, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=cm_norm, **pyplot_kwargs)
            lc.set_array(norm_attr[chan, :])
            plt_axis[chan].add_collection(lc)
            plt_axis[chan].set_ylim(
                1.2 * np.min(data[chan, :]), 1.2 * np.max(data[chan, :])
            )
            if channel_labels is not None:
                plt_axis[chan].set_ylabel(channel_labels[chan])

        plt.subplots_adjust(hspace=0)

    else:
        raise AssertionError("Invalid visualization method: {}".format(method))

    plt.xlim([x_values[0], x_values[-1]])

    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis[-1])
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.4)
        colorbar_alpha = alpha_overlay
        if (
            TimeseriesVisualizationMethod[method]
            == TimeseriesVisualizationMethod.colored_graph
        ):
            colorbar_alpha = 1.0
        plt_fig.colorbar(
            cm.ScalarMappable(cm_norm, cmap),
            orientation="horizontal",
            cax=colorbar_axis,
            alpha=colorbar_alpha,
        )
    if title:
        plt_axis[0].set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


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
        "true_class",
        "attr_class",
        "attr_score",
        "raw_input_ids",
        "convergence_score",
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_class,
        attr_score,
        raw_input_ids,
        convergence_score,
    ) -> None:
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input_ids = raw_input_ids
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


def visualize_text(
    datarecords: Iterable[VisualizationDataRecord], legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>True Label</th>"
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
                    format_classname(datarecord.true_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))
    display(html)

    return html
