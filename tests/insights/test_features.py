from unittest.mock import patch

import torch
from captum.insights.attr_vis.features import (
    EmptyFeature,
    FeatureOutput,
    GeneralFeature,
    ImageFeature,
    TextFeature,
    _convert_figure_base64,
)
from matplotlib.figure import Figure
from tests.helpers.basic import BaseTest


class TestTextFeature(BaseTest):
    FEATURE_NAME = "question"

    def test_text_feature_returns_text_as_visualization_type(self):
        feature = TextFeature(self.FEATURE_NAME, None, None, None)
        self.assertEqual(feature.visualization_type(), "text")

    def test_text_feature_uses_visualization_transform_if_provided(self):
        input_data = torch.rand(2, 2)
        transformed_data = torch.rand(1, 1)

        def mock_transform(data):
            return transformed_data

        feature = TextFeature(
            name=self.FEATURE_NAME,
            baseline_transforms=None,
            input_transforms=None,
            visualization_transform=mock_transform,
        )

        feature_output = feature.visualize(
            attribution=torch.rand(1, 1), data=input_data, contribution_frac=1.0
        )

        # has transformed data
        self.assertEqual(feature_output.base, transformed_data)

        feature = TextFeature(
            name=self.FEATURE_NAME,
            baseline_transforms=None,
            input_transforms=None,
            visualization_transform=None,
        )

        feature_output = feature.visualize(
            attribution=torch.rand(1, 1), data=input_data, contribution_frac=1.0
        )

        # has original data
        self.assertIs(feature_output.base, input_data)

    def test_text_feature_generates_correct_visualization_output(self):
        attribution = torch.tensor([0.1, 0.2, 0.3, 0.4])
        input_data = torch.rand(1, 2)
        expected_modified = [100 * x for x in (attribution / attribution.max())]
        contribution_frac = torch.rand(1).item()

        feature = TextFeature(
            name=self.FEATURE_NAME,
            baseline_transforms=None,
            input_transforms=None,
            visualization_transform=None,
        )

        feature_output = feature.visualize(attribution, input_data, contribution_frac)
        expected_feature_output = FeatureOutput(
            name=self.FEATURE_NAME,
            base=input_data,
            modified=expected_modified,
            type="text",
            contribution=contribution_frac,
        )

        self.assertEqual(expected_feature_output, feature_output)


class TestEmptyFeature(BaseTest):
    def test_empty_feature_should_generate_fixed_output(self):
        feature = EmptyFeature()
        contribution = torch.rand(1).item()
        expected_output = FeatureOutput(
            name="empty",
            base=None,
            modified=None,
            type="empty",
            contribution=contribution,
        )

        self.assertEqual(expected_output, feature.visualize(None, None, contribution))


class TestImageFeature(BaseTest):
    def test_image_feature_generates_correct_ouput(self):
        attribution = torch.zeros(1, 3, 4, 4)
        data = torch.ones(1, 3, 4, 4)
        contribution = 1.0
        name = "photo"

        orig_fig = Figure(figsize=(4, 4))
        attr_fig = Figure(figsize=(4, 4))

        def mock_viz_attr(*args, **kwargs):
            if kwargs["method"] == "original_image":
                return orig_fig, None
            else:
                return attr_fig, None

        feature = ImageFeature(
            name=name,
            baseline_transforms=None,
            input_transforms=None,
            visualization_transform=None,
        )

        with patch(
            "captum.attr._utils.visualization.visualize_image_attr", mock_viz_attr
        ):
            feature_output = feature.visualize(attribution, data, contribution)
            expected_feature_output = FeatureOutput(
                name=name,
                base=_convert_figure_base64(orig_fig),
                modified=_convert_figure_base64(attr_fig),
                type="image",
                contribution=contribution,
            )

            self.assertEqual(expected_feature_output, feature_output)


class TestGeneralFeature(BaseTest):
    def test_general_feature_generates_correct_output(self):
        name = "general_feature"
        categories = ["cat1", "cat2", "cat3", "cat4"]
        attribution = torch.Tensor(1, 4)
        attribution.fill_(0.5)
        data = torch.rand(1, 4)
        contribution = torch.rand(1).item()
        attr_squeezed = attribution.squeeze(0)

        expected_modified = [
            x * 100 for x in (attr_squeezed / attr_squeezed.norm()).tolist()
        ]
        expected_base = [
            f"{c}: {d:.2f}" for c, d in zip(categories, data.squeeze().tolist())
        ]

        feature = GeneralFeature(name, categories)

        feature_output = feature.visualize(
            attribution=attribution, data=data, contribution_frac=contribution
        )

        expected_feature_output = FeatureOutput(
            name=name,
            base=expected_base,
            modified=expected_modified,
            type="general",
            contribution=contribution,
        )

        self.assertEqual(expected_feature_output, feature_output)
