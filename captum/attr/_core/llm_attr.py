from copy import copy

from typing import Callable, cast, Dict, List, Optional, Union

import torch
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._utils.attribution import Attribution
from captum.attr._utils.interpretable_input import InterpretableInput, TextTemplateInput
from torch import nn, Tensor


SUPPORTED_METHODS = (FeatureAblation, ShapleyValueSampling, ShapleyValues)
SUPPORTED_INPUTS = (TextTemplateInput,)

DEFAULT_GEN_ARGS = {"max_new_tokens": 25, "do_sample": False}


class LLMAttributionResult:
    """
    Data class for the return result of LLMAttribution,
    which includes the necessary properties of the attribution.
    It also provides utilities to help present and plot the result in different forms.
    """

    def __init__(
        self,
        seq_attr: Tensor,
        token_attr: Tensor,
        input_tokens: List[str],
        output_tokens: List[str],
    ):
        self.seq_attr = seq_attr
        self.token_attr = token_attr
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def seq_attr_dict(self):
        return {k: v for v, k in zip(self.seq_attr.cpu().tolist(), self.input_tokens)}

    def plot_token_attr(self):
        pass

    def plot_seq_attr(self):
        pass


class LLMAttribution(Attribution):
    """
    Attribution class for large language models. It wraps a perturbation-based
    attribution algorthm to produce commonly interested attribution
    results for the use case of text generation.
    The wrapped instance will calculate attribution in the
    same way as configured in the original attribution algorthm, but it will provide a
    new "attribute" function which accepts text-based inputs
    and returns LLMAttributionResult
    """

    def __init__(
        self,
        attr_method: Attribution,
        tokenizer,
        attr_target: str = "log_prob",  # TODO: support callable attr_target
    ):
        """
        Args:
            attr_method (Attribution): instance of a supported perturbation attribution
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
            attr_target (str): attribute towards log probability or probability.
                    Available values ["log_prob", "prob"]
                    Default: "log_prob"
        """

        assert isinstance(
            attr_method, SUPPORTED_METHODS
        ), f"LLMAttribution does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method = copy(attr_method)

        self.attr_method.forward_func = self._forward_func

        # alias, we really need a model and don't support wrapper functions
        self.model = cast(nn.Module, self.forward_func)

        self.tokenizer = tokenizer
        self.device = (
            cast(torch.device, self.model.device)
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

        assert attr_target in (
            "log_prob",
            "prob",
        ), "attr_target should be either 'log_prob' or 'prob'"
        self.attr_target = attr_target

    def _forward_func(
        self,
        perturbed_feature,
        input_feature,
        target_tokens,
        _inspect_forward,
    ):
        perturbed_input = self._format_model_input(
            input_feature.to_model_input(perturbed_feature)
        )
        init_model_inp = perturbed_input

        model_inp = init_model_inp

        log_prob_list = []
        for target_token in target_tokens:
            output_logits = self.model.forward(
                model_inp, attention_mask=torch.tensor([[1] * model_inp.shape[1]])
            )
            new_token_logits = output_logits.logits[:, -1]
            log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)

            log_prob_list.append(log_probs[0][target_token].detach())

            model_inp = torch.cat(
                (model_inp, torch.tensor([[target_token]]).to(self.device)), dim=1
            )

        total_log_prob = sum(log_prob_list)
        # 1st element is the total prob, rest are the target tokens
        # add a leading dim for batch even we only support single instance for now
        target_log_probs = torch.stack(
            [total_log_prob, *log_prob_list], dim=0
        ).unsqueeze(0)
        target_probs = torch.exp(target_log_probs)

        if _inspect_forward:
            prompt = self.tokenizer.decode(init_model_inp[0])
            response = self.tokenizer.decode(target_tokens)

            # callback for externals to inspect (prompt, response, seq_prob)
            _inspect_forward(prompt, response, target_probs[0].tolist())

        return target_probs if self.attr_target != "log_prob" else target_log_probs

    def _format_model_input(self, model_input: Union[str, Tensor]):
        """
        Convert str to tokenized tensor
        to make LLMAttribution work with model inputs of both
        raw text and text token tensors
        """
        # return tensor(1, n_tokens)
        if isinstance(model_input, str):
            return self.tokenizer.encode(model_input, return_tensors="pt").to(
                self.device
            )
        return model_input.to(self.device)

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        num_trials: int = 1,
        gen_args: Optional[Dict] = None,
        # internal callback hook can be used for logging
        _inspect_forward: Optional[Callable] = None,
        **kwargs,
    ) -> LLMAttributionResult:
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            num_trials (int, optional): number of trials to run. Return is the average
                    attribibutions over all the trials.
                    Defaults: 1.
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_length": 25, "do_sample": False}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): attribution result
        """

        assert isinstance(
            inp, SUPPORTED_INPUTS
        ), f"LLMAttribution does not support input type {type(inp)}"

        if target is None:
            # generate when None
            assert hasattr(self.model, "generate") and callable(self.model.generate), (
                "The model does not have recognizable generate function."
                "Target must be given for attribution"
            )

            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS

            model_inp = self._format_model_input(inp.to_model_input())
            output_tokens = self.model.generate(model_inp, **gen_args)
            target_tokens = output_tokens[0][model_inp.size(1) :]
        else:
            assert gen_args is None, "gen_args must be None when target is given"

            if type(target) is str:
                # exclude sos
                target_tokens = self.tokenizer.encode(target)[1:]
            elif type(target) is torch.Tensor:
                target_tokens = target

        attr = torch.zeros(
            [1 + len(target_tokens), inp.n_itp_features],
            dtype=torch.float,
            device=self.device,
        )

        for _ in range(num_trials):
            attr_input = inp.to_tensor().to(self.device)

            cur_attr = self.attr_method.attribute(
                attr_input,
                additional_forward_args=(inp, target_tokens, _inspect_forward),
                **kwargs,
            )

            # temp necessary due to FA & Shapley's different return shape of multi-task
            # FA will flatten output shape internally (n_output_token, n_itp_features)
            # Shapley will keep output shape (batch, n_output_token, n_input_features)
            cur_attr = cur_attr.reshape(attr.shape)

            attr += cur_attr

        attr = attr / num_trials

        attr = inp.format_attr(attr)

        return LLMAttributionResult(
            attr[0],
            attr[1:],  # shape(n_output_token, n_input_features)
            inp.values,
            self.tokenizer.convert_ids_to_tokens(target_tokens),
        )
