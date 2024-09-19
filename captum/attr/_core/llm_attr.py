# pyre-strict
from copy import copy

from textwrap import shorten

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np

import torch
from captum._utils.typing import TokenizerLike
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.kernel_shap import KernelShap
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.lime import Lime
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._utils.attribution import (
    Attribution,
    GradientAttribution,
    PerturbationAttribution,
)
from captum.attr._utils.interpretable_input import (
    InterpretableInput,
    TextTemplateInput,
    TextTokenInput,
)
from torch import nn, Tensor

DEFAULT_GEN_ARGS: Dict[str, Any] = {
    "max_new_tokens": 25,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}


class LLMAttributionResult:
    """
    Data class for the return result of LLMAttribution,
    which includes the necessary properties of the attribution.
    It also provides utilities to help present and plot the result in different forms.
    """

    def __init__(
        self,
        seq_attr: Tensor,
        token_attr: Optional[Tensor],
        input_tokens: List[str],
        output_tokens: List[str],
    ) -> None:
        self.seq_attr = seq_attr
        self.token_attr = token_attr
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def seq_attr_dict(self) -> Dict[str, float]:
        return {k: v for v, k in zip(self.seq_attr.cpu().tolist(), self.input_tokens)}

    def plot_token_attr(
        self, show: bool = False
    ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
        """
        Generate a matplotlib plot for visualising the attribution
        of the output tokens.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False
        """

        if self.token_attr is None:
            raise ValueError(
                "token_attr is None (no token-level attribution was performed), please "
                "use plot_seq_attr instead for the sequence-level attribution plot"
            )
        token_attr = self.token_attr.cpu()  # type: ignore

        # maximum absolute attribution value
        # used as the boundary of normalization
        # always keep 0 as the mid point to differentiate pos/neg attr
        max_abs_attr_val = token_attr.abs().max().item()

        fig, ax = plt.subplots()

        # Hide the grid
        ax.grid(False)

        # Plot the heatmap
        data = token_attr.numpy()

        fig.set_size_inches(
            max(data.shape[1] * 1.3, 6.4), max(data.shape[0] / 2.5, 4.8)
        )
        colors = [
            "#93003a",
            "#d0365b",
            "#f57789",
            "#ffbdc3",
            "#ffffff",
            "#a4d6e1",
            "#73a3ca",
            "#4772b3",
            "#00429d",
        ]

        im = ax.imshow(
            data,
            vmax=max_abs_attr_val,
            vmin=-max_abs_attr_val,
            cmap=mcolors.LinearSegmentedColormap.from_list(
                name="colors", colors=colors
            ),
            aspect="auto",
        )
        fig.set_facecolor("white")

        # Create colorbar
        cbar = fig.colorbar(im, ax=ax)  # type: ignore
        cbar.ax.set_ylabel("Token Attribution", rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        shortened_tokens = [
            shorten(t, width=50, placeholder="...") for t in self.input_tokens
        ]
        ax.set_xticks(np.arange(data.shape[1]), labels=shortened_tokens)
        ax.set_yticks(np.arange(data.shape[0]), labels=self.output_tokens)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = "black" if 0.2 < im.norm(val) < 0.8 else "white"
                im.axes.text(
                    j,
                    i,
                    "%.4f" % val,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                )

        if show:
            plt.show()
            return None  # mypy wants this
        else:
            return fig, ax

    def plot_seq_attr(
        self, show: bool = False
    ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
        """
        Generate a matplotlib plot for visualising the attribution
        of the output sequence.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False
        """

        fig, ax = plt.subplots()

        data = self.seq_attr.cpu().numpy()

        fig.set_size_inches(max(data.shape[0] / 2, 6.4), max(data.shape[0] / 4, 4.8))

        shortened_tokens = [
            shorten(t, width=50, placeholder="...") for t in self.input_tokens
        ]
        ax.set_xticks(range(data.shape[0]), labels=shortened_tokens)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.setp(
            ax.get_xticklabels(),
            rotation=-30,
            ha="right",
            rotation_mode="anchor",
        )

        fig.set_facecolor("white")

        # pos bar
        ax.bar(
            range(data.shape[0]),
            [max(v, 0) for v in data],
            align="center",
            color="#4772b3",
        )
        # neg bar
        ax.bar(
            range(data.shape[0]),
            [min(v, 0) for v in data],
            align="center",
            color="#d0365b",
        )

        ax.set_ylabel("Sequence Attribution", rotation=90, va="bottom")

        if show:
            plt.show()
            return None  # mypy wants this
        else:
            return fig, ax


def _convert_ids_to_pretty_tokens(ids: Tensor, tokenizer: TokenizerLike) -> List[str]:
    """
    Convert ids to tokens without ugly unicode characters (e.g., Ġ). See:
    https://github.com/huggingface/transformers/issues/4786 and
    https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475/2

    This is the preferred function over tokenizer.convert_ids_to_tokens() for
    user-facing data.

    Quote from links:
    > Spaces are converted in a special character (the Ġ) in the tokenizer prior to
    > BPE splitting mostly to avoid digesting spaces since the standard BPE algorithm
    > used spaces in its process
    """
    return [tokenizer.decode(id_) for id_ in ids]


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

    SUPPORTED_METHODS = (
        FeatureAblation,
        ShapleyValueSampling,
        ShapleyValues,
        Lime,
        KernelShap,
    )
    SUPPORTED_PER_TOKEN_ATTR_METHODS = (
        FeatureAblation,
        ShapleyValueSampling,
        ShapleyValues,
    )
    SUPPORTED_INPUTS = (TextTemplateInput, TextTokenInput)

    def __init__(
        self,
        attr_method: PerturbationAttribution,
        tokenizer: TokenizerLike,
        attr_target: str = "log_prob",  # TODO: support callable attr_target
    ) -> None:
        """
        Args:
            attr_method (Attribution): Instance of a supported perturbation attribution
                    Supported methods include FeatureAblation, ShapleyValueSampling,
                    ShapleyValues, Lime, and KernelShap. Lime and KernelShap do not
                    support per-token attribution and will only return attribution
                    for the full target sequence.
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
            attr_target (str): attribute towards log probability or probability.
                    Available values ["log_prob", "prob"]
                    Default: "log_prob"
        """

        assert isinstance(
            attr_method, self.SUPPORTED_METHODS
        ), f"LLMAttribution does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method: PerturbationAttribution = copy(attr_method)
        self.include_per_token_attr: bool = isinstance(
            attr_method, self.SUPPORTED_PER_TOKEN_ATTR_METHODS
        )

        self.attr_method.forward_func = self._forward_func

        # alias, we really need a model and don't support wrapper functions
        # coz we need call model.forward, model.generate, etc.
        self.model: nn.Module = cast(nn.Module, self.forward_func)

        self.tokenizer: TokenizerLike = tokenizer
        self.device: torch.device = (
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
        perturbed_tensor: Union[None, Tensor],
        inp: InterpretableInput,
        target_tokens: Tensor,
        use_cached_outputs: bool = False,
        _inspect_forward: Optional[Callable[[str, str, List[float]], None]] = None,
    ) -> Tensor:
        # Lazily import transformers_typing to avoid importing transformers package if
        # it isn't needed
        from captum._utils.transformers_typing import (
            Cache,
            DynamicCache,
            supports_caching,
            update_model_kwargs,
        )

        perturbed_input = self._format_model_input(inp.to_model_input(perturbed_tensor))
        init_model_inp = perturbed_input

        model_inp = init_model_inp
        attention_mask = torch.ones(
            [1, model_inp.shape[1]], dtype=torch.long, device=model_inp.device
        )
        model_kwargs = {"attention_mask": attention_mask}
        # If applicable, update model kwargs for transformers models
        update_model_kwargs(
            model_kwargs=model_kwargs,
            model=self.model,
            input_ids=model_inp,
            caching=use_cached_outputs,
        )

        log_prob_list = []
        outputs = None
        for target_token in target_tokens:
            if use_cached_outputs:
                if outputs is not None:
                    # If applicable, convert past_key_values to DynamicCache for
                    # transformers models
                    if (
                        Cache is not None
                        and DynamicCache is not None
                        and supports_caching(self.model)
                        and not isinstance(outputs.past_key_values, Cache)
                    ):
                        outputs.past_key_values = DynamicCache.from_legacy_cache(
                            outputs.past_key_values
                        )
                    model_kwargs = self.model._update_model_kwargs_for_generation(
                        outputs, model_kwargs
                    )
                model_inputs = self.model.prepare_inputs_for_generation(
                    model_inp, **model_kwargs
                )
                outputs = self.model.forward(**model_inputs)
            else:
                # Update attention mask to adapt to input size change
                attention_mask = torch.ones(
                    [1, model_inp.shape[1]], dtype=torch.long, device=model_inp.device
                )
                model_kwargs["attention_mask"] = attention_mask
                outputs = self.model.forward(model_inp, **model_kwargs)
            new_token_logits = outputs.logits[:, -1]
            log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)

            log_prob_list.append(log_probs[0][target_token].detach())

            model_inp = torch.cat(
                (model_inp, torch.tensor([[target_token]]).to(self.device)), dim=1
            )

        # pyre-ignore[9] pyre/mypy thinks sum returns int here, but it will return
        # Tensor
        total_log_prob: Tensor = sum(log_prob_list)  # type: ignore
        # 1st element is the total prob, rest are the target tokens
        # add a leading dim for batch even we only support single instance for now
        if self.include_per_token_attr:
            target_log_probs = torch.stack(
                [total_log_prob, *log_prob_list], dim=0  # type: ignore
            ).unsqueeze(0)
        else:
            target_log_probs = total_log_prob  # type: ignore
        target_probs = torch.exp(target_log_probs)

        if _inspect_forward:
            prompt = self.tokenizer.decode(init_model_inp[0])
            response = self.tokenizer.decode(target_tokens)

            # callback for externals to inspect (prompt, response, seq_prob)
            _inspect_forward(prompt, response, target_probs[0].tolist())

        return target_probs if self.attr_target != "log_prob" else target_log_probs

    def _format_model_input(self, model_input: Union[str, Tensor]) -> Tensor:
        """
        Convert str to tokenized tensor
        to make LLMAttribution work with model inputs of both
        raw text and text token tensors
        """
        # return tensor(1, n_tokens)
        if isinstance(model_input, str):
            # pyre-ignore[9] pyre/mypy thinks return type may be List, but it will be
            # Tensor
            return self.tokenizer.encode(  # type: ignore
                model_input, return_tensors="pt"
            ).to(self.device)
        return model_input.to(self.device)

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        skip_tokens: Union[List[int], List[str], None] = None,
        num_trials: int = 1,
        gen_args: Optional[Dict[str, Any]] = None,
        use_cached_outputs: bool = True,
        # internal callback hook can be used for logging
        _inspect_forward: Optional[Callable[[str, str, List[float]], None]] = None,
        **kwargs: Any,
    ) -> LLMAttributionResult:
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            skip_tokens (List[int] or List[str], optional): the tokens to skip in the
                    the output's interpretable representation. Use this argument to
                    define uninterested tokens, commonly like special tokens, e.g.,
                    sos, and unk. It can be a list of strings of the tokens or a list
                    of integers of the token ids.
                    Default: None
            num_trials (int, optional): number of trials to run. Return is the average
                    attribibutions over all the trials.
                    Defaults: 1.
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_new_tokens": 25, "do_sample": False,
                    "temperature": None, "top_p": None}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): Attribution result. token_attr will be None
                    if attr method is Lime or KernelShap.
        """

        assert isinstance(
            inp, self.SUPPORTED_INPUTS
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
            # Encode skip tokens
            if skip_tokens:
                if isinstance(skip_tokens[0], str):
                    skip_tokens = cast(List[str], skip_tokens)
                    skip_tokens = self.tokenizer.convert_tokens_to_ids(skip_tokens)
            else:
                skip_tokens = []

            if isinstance(target, str):
                encoded = self.tokenizer.encode(target)
                target_tokens = torch.tensor(
                    [token for token in encoded if token not in skip_tokens]
                )
            elif isinstance(target, torch.Tensor):
                target_tokens = target[
                    ~torch.isin(target, torch.tensor(skip_tokens, device=target.device))
                ]
            else:
                raise TypeError(
                    "target must either be str or Tensor, but the type of target is "
                    "{}".format(type(target))
                )

        attr = torch.zeros(
            [
                1 + len(target_tokens) if self.include_per_token_attr else 1,
                inp.n_itp_features,
            ],
            dtype=torch.float,
            device=self.device,
        )

        for _ in range(num_trials):
            attr_input = inp.to_tensor().to(self.device)

            cur_attr = self.attr_method.attribute(
                attr_input,
                additional_forward_args=(
                    inp,
                    target_tokens,
                    use_cached_outputs,
                    _inspect_forward,
                ),
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
            (
                attr[1:] if self.include_per_token_attr else None
            ),  # shape(n_output_token, n_input_features)
            inp.values,
            _convert_ids_to_pretty_tokens(target_tokens, self.tokenizer),
        )

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        r"""
        This method is not implemented for LLMAttribution.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for LLMAttribution"
        )


class LLMGradientAttribution(Attribution):
    """
    Attribution class for large language models. It wraps a gradient-based
    attribution algorthm to produce commonly interested attribution
    results for the use case of text generation.
    The wrapped instance will calculate attribution in the
    same way as configured in the original attribution algorthm,
    with respect to the log probabilities of each
    generated token and the whole sequence. It will provide a
    new "attribute" function which accepts text-based inputs
    and returns LLMAttributionResult
    """

    SUPPORTED_METHODS = (
        LayerGradientShap,
        LayerGradientXActivation,
        LayerIntegratedGradients,
    )
    SUPPORTED_INPUTS = (TextTokenInput,)

    def __init__(
        self,
        attr_method: GradientAttribution,
        tokenizer: TokenizerLike,
    ) -> None:
        """
        Args:
            attr_method (Attribution): instance of a supported perturbation attribution
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
        """
        assert isinstance(
            attr_method, self.SUPPORTED_METHODS
        ), f"LLMGradientAttribution does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # alias, we really need a model and don't support wrapper functions
        # coz we need call model.forward, model.generate, etc.
        self.model: nn.Module = cast(nn.Module, self.forward_func)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method: GradientAttribution = copy(attr_method)
        self.attr_method.forward_func = GradientForwardFunc(self)

        self.tokenizer: TokenizerLike = tokenizer
        self.device: torch.device = (
            cast(torch.device, self.model.device)
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

    def _format_model_input(self, model_input: Tensor) -> Tensor:
        """
        Convert str to tokenized tensor
        """
        return model_input.to(self.device)

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        skip_tokens: Union[List[int], List[str], None] = None,
        gen_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMAttributionResult:
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            skip_tokens (List[int] or List[str], optional): the tokens to skip in the
                    the output's interpretable representation. Use this argument to
                    define uninterested tokens, commonly like special tokens, e.g.,
                    sos, and unk. It can be a list of strings of the tokens or a list
                    of integers of the token ids.
                    Default: None
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_new_tokens": 25, "do_sample": False,
                    "temperature": None, "top_p": None}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): attribution result
        """

        assert isinstance(
            inp, self.SUPPORTED_INPUTS
        ), f"LLMGradAttribution does not support input type {type(inp)}"

        if target is None:
            # generate when None
            assert hasattr(self.model, "generate") and callable(self.model.generate), (
                "The model does not have recognizable generate function."
                "Target must be given for attribution"
            )

            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS

            with torch.no_grad():
                model_inp = self._format_model_input(inp.to_model_input())
                output_tokens = self.model.generate(model_inp, **gen_args)
                target_tokens = output_tokens[0][model_inp.size(1) :]
        else:
            assert gen_args is None, "gen_args must be None when target is given"
            # Encode skip tokens
            if skip_tokens:
                if isinstance(skip_tokens[0], str):
                    skip_tokens = cast(List[str], skip_tokens)
                    skip_tokens = self.tokenizer.convert_tokens_to_ids(skip_tokens)
            else:
                skip_tokens = []

            if isinstance(target, str):
                encoded = self.tokenizer.encode(target)
                target_tokens = torch.tensor(
                    [token for token in encoded if token not in skip_tokens]
                )
            elif isinstance(target, torch.Tensor):
                target_tokens = target[
                    ~torch.isin(target, torch.tensor(skip_tokens, device=target.device))
                ]
            else:
                raise TypeError(
                    "target must either be str or Tensor, but the type of target is "
                    "{}".format(type(target))
                )

        attr_inp = inp.to_tensor().to(self.device)

        attr_list = []
        for cur_target_idx, _ in enumerate(target_tokens):
            # attr in shape(batch_size, input+output_len, emb_dim)
            attr = self.attr_method.attribute(
                attr_inp,
                additional_forward_args=(
                    inp,
                    target_tokens,
                    cur_target_idx,
                ),
                **kwargs,
            ).detach()
            attr = cast(Tensor, attr)

            # will have the attr for previous output tokens
            # cut to shape(batch_size, inp_len, emb_dim)
            if cur_target_idx:
                attr = attr[:, :-cur_target_idx]

            # the author of IG uses sum
            # https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L350
            attr = attr.sum(-1)

            attr_list.append(attr)

        # assume inp batch only has one instance
        # to shape(n_output_token, ...)
        attr = torch.cat(attr_list, dim=0)

        # grad attr method do not care the length of features in interpretable format
        # it attributes to all the elements of the output of the specified layer
        # so we need special handling for the inp type which don't care all the elements
        if isinstance(inp, TextTokenInput) and inp.itp_mask is not None:
            itp_mask = inp.itp_mask.to(attr.device)
            itp_mask = itp_mask.expand_as(attr)
            attr = attr[itp_mask].view(attr.size(0), -1)

        # for all the gradient methods we support in this class
        # the seq attr is the sum of all the token attr if the attr_target is log_prob,
        # shape(n_input_features)
        seq_attr = attr.sum(0)

        return LLMAttributionResult(
            seq_attr,
            attr,  # shape(n_output_token, n_input_features)
            inp.values,
            _convert_ids_to_pretty_tokens(target_tokens, self.tokenizer),
        )

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        r"""
        This method is not implemented for LLMGradientAttribution.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for LLMGradientAttribution"
        )


class GradientForwardFunc(nn.Module):
    """
    A wrapper class for the forward function of a model in LLMGradientAttribution
    """

    def __init__(self, attr: LLMGradientAttribution) -> None:
        super().__init__()
        self.attr = attr
        self.model: nn.Module = attr.model

    def forward(
        self,
        perturbed_tensor: Tensor,
        inp: InterpretableInput,
        target_tokens: Tensor,  # 1D tensor of target token ids
        cur_target_idx: int,  # current target index
    ) -> Tensor:
        perturbed_input = self.attr._format_model_input(
            inp.to_model_input(perturbed_tensor)
        )

        if cur_target_idx:
            # the input batch size can be expanded by attr method
            output_token_tensor = (
                target_tokens[:cur_target_idx]
                .unsqueeze(0)
                .expand(perturbed_input.size(0), -1)
                .to(self.attr.device)
            )
            new_input_tensor = torch.cat([perturbed_input, output_token_tensor], dim=1)
        else:
            new_input_tensor = perturbed_input

        output_logits = self.model(new_input_tensor)

        new_token_logits = output_logits.logits[:, -1]
        log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)

        target_token = target_tokens[cur_target_idx]
        token_log_probs = log_probs[..., target_token]

        # the attribution target is limited to the log probability
        return token_log_probs
