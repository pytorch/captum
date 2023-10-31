#!/usr/bin/env python3

import torch
from captum.attr._utils.interpretable_input import TextTemplateInput
from parameterized import parameterized
from tests.helpers.basic import assertTensorAlmostEqual, BaseTest


class TestTextTemplateInput(BaseTest):
    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
            ),
        ]
    )
    def test_input(self, template, values) -> None:
        tt_input = TextTemplateInput(template, values)

        expected_tensor = torch.tensor([[1.0] * 4])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        self.assertEqual(tt_input.to_model_input(), "a b c d e f")

        perturbed_tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        self.assertEqual(tt_input.to_model_input(perturbed_tensor), "a b  d e ")

    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"], ["w", "x", "y", "z"]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
                {"arg1": "w", "arg2": "x", "arg3": "y", "arg4": "z"},
            ),
        ]
    )
    def test_input_with_baselines(self, template, values, baselines) -> None:
        perturbed_tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        # single instance baselines
        tt_input = TextTemplateInput(template, values, baselines=baselines)
        self.assertEqual(tt_input.to_model_input(perturbed_tensor), "a b x d e z")

    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"], [0, 1, 0, 1]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
                {"arg1": 0, "arg2": 1, "arg3": 0, "arg4": 1},
            ),
        ]
    )
    def test_input_with_mask(self, template, values, mask) -> None:
        tt_input = TextTemplateInput(template, values, mask=mask)

        expected_tensor = torch.tensor([[1.0] * 2])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        self.assertEqual(tt_input.to_model_input(), "a b c d e f")

        perturbed_tensor = torch.tensor([[1.0, 0.0]])
        self.assertEqual(tt_input.to_model_input(perturbed_tensor), "a b  d e ")

    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"], [0, 1, 0, 1]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
                {"arg1": 0, "arg2": 1, "arg3": 0, "arg4": 1},
            ),
        ]
    )
    def test_format_attr(self, template, values, mask) -> None:
        tt_input = TextTemplateInput(template, values, mask=mask)

        attr = torch.tensor([[0.1, 0.2]])

        assertTensorAlmostEqual(
            self, tt_input.format_attr(attr), torch.tensor([[0.1, 0.2, 0.1, 0.2]])
        )
