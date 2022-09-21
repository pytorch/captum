#!/usr/bin/env python3
import base64
import warnings
from collections import namedtuple
from io import BytesIO
from typing import Callable, List, Optional, Union

from captum._utils.common import safe_div
from captum.attr._utils import visualization as viz
from captum.insights.attr_vis._utils.transforms import format_transforms

FeatureOutput = namedtuple("FeatureOutput", "name base modified type contribution")


def _convert_figure_base64(fig):
    buff = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()  # removes padding
    fig.savefig(buff, format="png")
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


class BaseFeature:
    r"""
    All Feature classes extend this class to implement custom visualizations in
    Insights.

    It enforces child classes to implement ``visualization_type`` and ``visualize``
    methods.
    """

    def __init__(
        self,
        name: str,
        baseline_transforms: Optional[Union[Callable, List[Callable]]],
        input_transforms: Optional[Union[Callable, List[Callable]]],
        visualization_transform: Optional[Callable],
    ) -> None:
        r"""
        Args:

            name (str): The label of the specific feature. For example, an
                        ImageFeature's name can be "Photo".
            baseline_transforms (list, Callable, optional): Optional list of
                        callables (e.g. functions) to be called on the input tensor
                        to construct multiple baselines. Currently only one baseline
                        is supported. See
                        :py:class:`.IntegratedGradients` for more
                        information about baselines.
            input_transforms (list, Callable, optional): Optional list of callables
                        (e.g. functions) called on the input tensor sequentially to
                        convert it into the format expected by the model.
            visualization_transform (Callable, optional): Optional callable (e.g.
                        function) applied as a postprocessing step of the original
                        input data (before ``input_transforms``) to convert it to a
                        format to be understood by the frontend visualizer as
                        specified in ``captum/captum/insights/frontend/App.js``.
        """
        self.name = name
        self.baseline_transforms = format_transforms(baseline_transforms)
        self.input_transforms = format_transforms(input_transforms)
        self.visualization_transform = visualization_transform

    @staticmethod
    def visualization_type() -> str:
        raise NotImplementedError

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        raise NotImplementedError


class ImageFeature(BaseFeature):
    r"""
    ImageFeature is used to visualize image features in Insights. It expects an image in
    NCHW format. If C has a dimension of 1, its assumed to be a greyscale image.
    If it has a dimension of 3, its expected to be in RGB format.
    """

    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Callable, List[Callable]],
        input_transforms: Union[Callable, List[Callable]],
        visualization_transform: Optional[Callable] = None,
    ) -> None:
        r"""
        Args:
            name (str): The label of the specific feature. For example, an
                        ImageFeature's name can be "Photo".
            baseline_transforms (list, Callable, optional): Optional list of
                        callables (e.g. functions) to be called on the input tensor
                        to construct multiple baselines. Currently only one baseline
                        is supported. See
                        :py:class:`.IntegratedGradients` for more
                        information about baselines.
            input_transforms (list, Callable, optional): A list of transforms
                        or transform to be applied to the input. For images,
                        normalization is often applied here.
            visualization_transform (Callable, optional): Optional callable (e.g.
                        function) applied as a postprocessing step of the original
                        input data (before input_transforms) to convert it to a
                        format to be visualized.
        """
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=visualization_transform,
        )

    @staticmethod
    def visualization_type() -> str:
        return "image"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        if self.visualization_transform:
            data = self.visualization_transform(data)

        data_t, attribution_t = [
            t.detach().squeeze().permute((1, 2, 0)).cpu().numpy()
            for t in (data, attribution)
        ]

        orig_fig, _ = viz.visualize_image_attr(
            attribution_t, data_t, method="original_image", use_pyplot=False
        )
        attr_fig, _ = viz.visualize_image_attr(
            attribution_t,
            data_t,
            method="heat_map",
            sign="absolute_value",
            use_pyplot=False,
        )

        img_64 = _convert_figure_base64(orig_fig)
        attr_img_64 = _convert_figure_base64(attr_fig)

        return FeatureOutput(
            name=self.name,
            base=img_64,
            modified=attr_img_64,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )


class TextFeature(BaseFeature):
    r"""
    TextFeature is used to visualize text (e.g. sentences) in Insights.
    It expects the visualization transform to convert the input data (e.g. index to
    string) to the raw text.
    """

    def __init__(
        self,
        name: str,
        baseline_transforms: Union[Callable, List[Callable]],
        input_transforms: Union[Callable, List[Callable]],
        visualization_transform: Callable,
    ) -> None:
        r"""
        Args:
            name (str): The label of the specific feature. For example, an
                        ImageFeature's name can be "Photo".
            baseline_transforms (list, Callable, optional): Optional list of
                        callables (e.g. functions) to be called on the input tensor
                        to construct multiple baselines. Currently only one baseline
                        is supported. See
                        :py:class:`.IntegratedGradients` for more
                        information about baselines.
                        For text features, a common baseline is a tensor of indices
                        corresponding to PAD with the same size as the input
                        tensor. See :py:class:`.TokenReferenceBase` for more
                        information.
            input_transforms (list, Callable, optional): A list of transforms
                        or transform to be applied to the input. For text, a common
                        transform is to convert the tokenized input tensor into an
                        interpretable embedding. See
                        :py:class:`.InterpretableEmbeddingBase`
                        and
                        :py:func:`~.configure_interpretable_embedding_layer`
                        for more information.
            visualization_transform (Callable, optional): Optional callable (e.g.
                        function) applied as a postprocessing step of the original
                        input data (before ``input_transforms``) to convert it to a
                        suitable format for visualization. For text features,
                        a common function is to convert the token indices to their
                        corresponding (sub)words.
        """
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=visualization_transform,
        )

    @staticmethod
    def visualization_type() -> str:
        return "text"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        if self.visualization_transform:
            text = self.visualization_transform(data)
        else:
            text = data

        attribution = attribution.squeeze(0)
        data = data.squeeze(0)
        if len(attribution.shape) > 1:
            attribution = attribution.sum(dim=1)

        # L-Infinity norm, if norm is 0, all attr elements are 0
        attr_max = attribution.abs().max()
        normalized_attribution = safe_div(attribution, attr_max)

        modified = [x * 100 for x in normalized_attribution.tolist()]
        return FeatureOutput(
            name=self.name,
            base=text,
            modified=modified,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )


class GeneralFeature(BaseFeature):
    r"""
    GeneralFeature is used for non-specified feature visualization in Insights.
    It can be used for dense or sparse features.

    Currently general features are only supported for 2-d tensors, in the format (N, C)
    where N is the number of samples and C is the number of categories.
    """

    def __init__(self, name: str, categories: List[str]) -> None:
        r"""
        Args:
            name (str): The label of the specific feature. For example, an
                        ImageFeature's name can be "Photo".
            categories (list[str]): Category labels for the general feature. The
                        order and size should match the second dimension of the
                        ``data`` tensor parameter in ``visualize``.
        """
        super().__init__(
            name,
            baseline_transforms=None,
            input_transforms=None,
            visualization_transform=None,
        )
        self.categories = categories

    @staticmethod
    def visualization_type() -> str:
        return "general"

    def visualize(self, attribution, data, contribution_frac) -> FeatureOutput:
        attribution = attribution.squeeze(0)
        data = data.squeeze(0)

        # L-2 norm, if norm is 0, all attr elements are 0
        l2_norm = attribution.norm()
        normalized_attribution = safe_div(attribution, l2_norm)

        modified = [x * 100 for x in normalized_attribution.tolist()]

        base = [f"{c}: {d:.2f}" for c, d in zip(self.categories, data.tolist())]
        return FeatureOutput(
            name=self.name,
            base=base,
            modified=modified,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )


class EmptyFeature(BaseFeature):
    def __init__(
        self,
        name: str = "empty",
        baseline_transforms: Optional[Union[Callable, List[Callable]]] = None,
        input_transforms: Optional[Union[Callable, List[Callable]]] = None,
        visualization_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            name,
            baseline_transforms=baseline_transforms,
            input_transforms=input_transforms,
            visualization_transform=visualization_transform,
        )

    @staticmethod
    def visualization_type() -> str:
        return "empty"

    def visualize(self, _attribution, _data, contribution_frac) -> FeatureOutput:
        return FeatureOutput(
            name=self.name,
            base=None,
            modified=None,
            type=self.visualization_type(),
            contribution=contribution_frac,
        )
