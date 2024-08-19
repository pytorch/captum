#!/usr/bin/env python3

# pyre-unsafe

import torch
from captum.attr._utils.interpretable_input import TextTemplateInput, TextTokenInput
from parameterized import parameterized
from tests.helpers import BaseTest
from tests.helpers.basic import assertTensorAlmostEqual
from torch import Tensor


class DummyTokenizer:
    def __init__(self, vocab_list) -> None:
        self.token_to_id = {v: i for i, v in enumerate(vocab_list)}
        self.id_to_token = vocab_list
        self.unk_idx = len(vocab_list) + 1

    def encode(self, text, **kwargs) -> Tensor:
        return torch.tensor([self.convert_tokens_to_ids(text.split(" "))])

    def convert_ids_to_tokens(self, ids):
        return [
            (self.id_to_token[i] if i < len(self.id_to_token) else "[UNK]") for i in ids
        ]

    def convert_tokens_to_ids(self, tokens):
        return [
            (self.token_to_id[t] if t in self.token_to_id else self.unk_idx)
            for t in tokens
        ]


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


class TestTextTokenInput(BaseTest):
    def test_input(self) -> None:
        tokenizer = DummyTokenizer(["a", "b", "c"])
        tt_input = TextTokenInput("a c d", tokenizer)

        expected_tensor = torch.tensor([[1.0] * 3])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        expected_model_inp = torch.tensor([[0, 2, tokenizer.unk_idx]])
        assertTensorAlmostEqual(self, tt_input.to_model_input(), expected_model_inp)

        perturbed_tensor = torch.tensor([[1.0, 0.0, 0.0]])
        expected_perturbed_inp = torch.tensor([[0, 0, 0]])
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

    def test_input_with_baselines(self) -> None:
        tokenizer = DummyTokenizer(["a", "b", "c"])

        # int baselines
        tt_input = TextTokenInput("a c d", tokenizer, baselines=1)

        perturbed_tensor = torch.tensor([[1.0, 0.0, 0.0]])
        expected_perturbed_inp = torch.tensor([[0, 1, 1]])
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

        # str baselines
        tt_input = TextTokenInput("a c d", tokenizer, baselines="b")
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

    def test_input_with_skip_tokens(self) -> None:
        tokenizer = DummyTokenizer(["a", "b", "c"])

        # int skip tokens
        tt_input = TextTokenInput("a c d", tokenizer, skip_tokens=[0])

        expected_tensor = torch.tensor([[1.0] * 2])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        expected_model_inp = torch.tensor([[0, 2, tokenizer.unk_idx]])
        assertTensorAlmostEqual(self, tt_input.to_model_input(), expected_model_inp)

        perturbed_tensor = torch.tensor([[0.0, 0.0]])
        expected_perturbed_inp = torch.tensor([[0, 0, 0]])
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

        # str skip tokens
        tt_input = TextTokenInput("a c d", tokenizer, skip_tokens=["a"])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)
        assertTensorAlmostEqual(self, tt_input.to_model_input(), expected_model_inp)
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )
