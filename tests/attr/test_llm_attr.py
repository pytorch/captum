#!/usr/bin/env python3

import copy
from collections import namedtuple
from typing import Any, cast, Dict, List, Optional, Union

import torch
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.kernel_shap import KernelShap
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.lime import Lime
from captum.attr._core.llm_attr import LLMAttribution, LLMGradientAttribution
from captum.attr._core.shapley_value import ShapleyValueSampling
from captum.attr._utils.interpretable_input import TextTemplateInput, TextTokenInput
from parameterized import parameterized, parameterized_class
from tests.helpers import BaseTest
from tests.helpers.basic import assertTensorAlmostEqual
from torch import nn, Tensor


class DummyTokenizer:
    vocab_size = 256
    sos, unk = [0, 1]
    special_tokens = {sos: "<sos>", unk: "<unk>"}

    def encode(
        self, text: str, return_tensors: Optional[str] = None
    ) -> Union[List[int], Tensor]:
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

    def decode(self, token_ids) -> str:
        return " ".join(self.convert_ids_to_tokens(token_ids))


class DummyLLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = DummyTokenizer()
        self.emb = nn.Embedding(self.tokenizer.vocab_size, 10)
        self.linear = nn.Linear(10, self.tokenizer.vocab_size)
        self.trans = nn.TransformerEncoderLayer(d_model=10, nhead=2)

    def forward(self, input_ids, *args, **kwargs) -> Any:
        emb = self.emb(input_ids)
        if "past_key_values" in kwargs:
            emb = torch.cat((kwargs["past_key_values"], emb), dim=1)
        logits = self.linear(self.trans(emb))
        Result = namedtuple("Result", ["logits", "past_key_values"])
        return Result(logits=logits, past_key_values=emb)

    def generate(
        self, input_ids, *args, mock_response: Optional[str] = None, **kwargs
    ) -> Tensor:
        assert mock_response, "must mock response to use DummyLLM to geenrate"
        response = self.tokenizer.encode(mock_response)[1:]
        return torch.cat(
            [input_ids, torch.tensor([response], device=self.device)], dim=1
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs):
        new_kwargs = copy.deepcopy(model_kwargs)
        if hasattr(outputs, "past_key_values"):
            new_kwargs["past_key_values"] = outputs.past_key_values
        return new_kwargs

    def prepare_inputs_for_generation(
        self, model_inp, **model_kwargs
    ) -> Dict[str, Any]:
        if "past_key_values" in model_kwargs:
            emb_len = model_kwargs["past_key_values"].shape[1]
            return {
                "input_ids": model_inp[:, emb_len:],
                "past_key_values": model_kwargs["past_key_values"],
            }
        return {"input_ids": model_inp}

    @property
    def device(self) -> Any:
        return next(self.parameters()).device


@parameterized_class(
    ("device", "use_cached_outputs"),
    (
        [("cpu", True), ("cpu", False), ("cuda", True), ("cuda", False)]
        if torch.cuda.is_available()
        else [("cpu", True), ("cpu", False)]
    ),
)
class TestLLMAttr(BaseTest):
    device: str
    use_cached_outputs: bool

    @parameterized.expand([(FeatureAblation,), (ShapleyValueSampling,)])
    def test_llm_attr(self, AttrClass) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        llm_attr = LLMAttribution(AttrClass(llm), tokenizer)

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_attr.attribute(
            inp, "m n o p q", use_cached_outputs=self.use_cached_outputs
        )

        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(cast(Tensor, res.token_attr).shape, (5, 4))
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(res.output_tokens, ["m", "n", "o", "p", "q"])
        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(cast(Tensor, res.token_attr).device.type, self.device)

    def test_llm_attr_without_target(self) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        fa = FeatureAblation(llm)
        llm_fa = LLMAttribution(fa, tokenizer)

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_fa.attribute(
            inp,
            gen_args={"mock_response": "x y z"},
            use_cached_outputs=self.use_cached_outputs,
        )

        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(cast(Tensor, res.token_attr).shape, (3, 4))
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(res.output_tokens, ["x", "y", "z"])
        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(cast(Tensor, res.token_attr).device.type, self.device)

    def test_llm_attr_fa_log_prob(self) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        fa = FeatureAblation(llm)
        llm_fa = LLMAttribution(fa, tokenizer, attr_target="log_prob")

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_fa.attribute(
            inp, "m n o p q", use_cached_outputs=self.use_cached_outputs
        )

        # With FeatureAblation, the seq attr in log_prob
        # equals to the sum of each token attr
        assertTensorAlmostEqual(self, res.seq_attr, cast(Tensor, res.token_attr).sum(0))

    @parameterized.expand([(Lime,), (KernelShap,)])
    def test_llm_attr_without_token(self, AttrClass) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        fa = AttrClass(llm)
        llm_fa = LLMAttribution(fa, tokenizer, attr_target="log_prob")

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_fa.attribute(
            inp, "m n o p q", use_cached_outputs=self.use_cached_outputs
        )

        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(res.token_attr, None)
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(res.output_tokens, ["m", "n", "o", "p", "q"])


@parameterized_class(
    ("device",), [("cpu",), ("cuda",)] if torch.cuda.is_available() else [("cpu",)]
)
class TestLLMGradAttr(BaseTest):
    device: str

    def test_llm_attr(self) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        attr = LayerIntegratedGradients(llm, llm.emb)
        llm_attr = LLMGradientAttribution(attr, tokenizer)

        inp = TextTokenInput("a b c", tokenizer)
        res = llm_attr.attribute(inp, "m n o p q")

        # 5 output tokens, 4 input tokens including sos
        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(res.token_attr.shape, (5, 4))
        self.assertEqual(res.input_tokens, ["<sos>", "a", "b", "c"])
        self.assertEqual(res.output_tokens, ["m", "n", "o", "p", "q"])

        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(res.token_attr.device.type, self.device)

    def test_llm_attr_without_target(self) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        attr = LayerIntegratedGradients(llm, llm.emb)
        llm_attr = LLMGradientAttribution(attr, tokenizer)

        inp = TextTokenInput("a b c", tokenizer)
        res = llm_attr.attribute(inp, gen_args={"mock_response": "x y z"})

        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(res.token_attr.shape, (3, 4))
        self.assertEqual(res.input_tokens, ["<sos>", "a", "b", "c"])
        self.assertEqual(res.output_tokens, ["x", "y", "z"])

        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(res.token_attr.device.type, self.device)

    def test_llm_attr_with_skip_tokens(self) -> None:
        llm = DummyLLM()
        llm.to(self.device)
        tokenizer = DummyTokenizer()
        attr = LayerIntegratedGradients(llm, llm.emb)
        llm_attr = LLMGradientAttribution(attr, tokenizer)

        inp = TextTokenInput("a b c", tokenizer, skip_tokens=[0])
        res = llm_attr.attribute(inp, "m n o p q")

        # 5 output tokens, 4 input tokens including sos
        self.assertEqual(res.seq_attr.shape, (3,))
        self.assertEqual(res.token_attr.shape, (5, 3))
        self.assertEqual(res.input_tokens, ["a", "b", "c"])
        self.assertEqual(res.output_tokens, ["m", "n", "o", "p", "q"])

        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(res.token_attr.device.type, self.device)
