#!/usr/bin/env python3

from collections import namedtuple
from typing import List, Optional, Union

import torch
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.llm_attr import LLMAttribution
from captum.attr._core.shapley_value import ShapleyValueSampling
from captum.attr._utils.interpretable_input import TextTemplateInput
from parameterized import parameterized
from tests.helpers.basic import assertTensorAlmostEqual, BaseTest
from torch import nn, Tensor


class DummyTokenizer:
    vocab_size = 256
    sos, unk = [0, 1]
    special_tokens = {sos: "<sos>", unk: "<unk>"}

    def encode(self, text: str, return_tensors: Optional[str] = None):
        tokens = text.split(" ")
        tokens_ids = [ord(s[0]) if len(s) == 1 else self.unk for s in tokens]

        # start with sos
        tokens_ids: Union[List[int], Tensor] = [self.sos, *tokens_ids]

        if return_tensors:
            tokens_ids = torch.tensor([tokens_ids])

        return tokens_ids

    def convert_ids_to_tokens(self, token_ids):
        return [
            (self.special_tokens[tid] if tid in self.special_tokens else chr(tid))
            for tid in token_ids
        ]

    def decode(self, token_ids):
        return " ".join(self.convert_ids_to_tokens(token_ids))


class DummyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DummyTokenizer()
        self.emb = nn.Embedding(self.tokenizer.vocab_size, 10)
        self.linear = nn.Linear(10, self.tokenizer.vocab_size)
        self.trans = nn.TransformerEncoderLayer(d_model=10, nhead=2)

    def forward(self, input_ids, *args, **kwargs):
        emb = self.emb(input_ids)
        logits = self.linear(self.trans(emb))
        Result = namedtuple("Result", ["logits"])
        return Result(logits=logits)

    def generate(self, input_ids, *args, mock_response=None, **kwargs):
        assert mock_response, "must mock response to use DummyLLM to geenrate"
        response = self.tokenizer.encode(mock_response)[1:]
        return torch.cat([input_ids, torch.tensor([response])], dim=1)

    @property
    def device(self):
        return next(self.parameters()).device


class TestLLMAttr(BaseTest):
    @parameterized.expand([(FeatureAblation,), (ShapleyValueSampling,)])
    def test_llm_attr(self, AttrClass) -> None:
        llm = DummyLLM()
        tokenizer = DummyTokenizer()
        llm_attr = LLMAttribution(AttrClass(llm), tokenizer)

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_attr.attribute(inp, "m n o p q")

        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(res.token_attr.shape, (5, 4))
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(res.output_tokens, ["m", "n", "o", "p", "q"])

    def test_llm_attr_without_target(self) -> None:
        llm = DummyLLM()
        tokenizer = DummyTokenizer()
        fa = FeatureAblation(llm)
        llm_fa = LLMAttribution(fa, tokenizer)

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_fa.attribute(inp, gen_args={"mock_response": "x y z"})

        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(res.token_attr.shape, (3, 4))
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(res.output_tokens, ["x", "y", "z"])

    def test_llm_attr_fa_log_prob(self) -> None:
        llm = DummyLLM()
        tokenizer = DummyTokenizer()
        fa = FeatureAblation(llm)
        llm_fa = LLMAttribution(fa, tokenizer, attr_target="log_prob")

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_fa.attribute(inp, "m n o p q")

        # With FeatureAblation, the seq attr in log_prob
        # equals to the sum of each token attr
        assertTensorAlmostEqual(self, res.seq_attr, res.token_attr.sum(0))
