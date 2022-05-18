#!/usr/bin/env python3
import inspect
from collections import namedtuple
from typing import (
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from captum._utils.common import _run_forward, safe_div
from captum.insights.attr_vis.config import (
    ATTRIBUTION_METHOD_CONFIG,
    ATTRIBUTION_NAMES_TO_METHODS,
)
from captum.insights.attr_vis.features import BaseFeature
from torch import Tensor
from torch.nn import Module

OutputScore = namedtuple("OutputScore", "score index label")


class AttributionCalculation:
    def __init__(
        self,
        models: Sequence[Module],
        classes: Sequence[str],
        features: List[BaseFeature],
        score_func: Optional[Callable] = None,
        use_label_for_attr: bool = True,
    ) -> None:
        self.models = models
        self.classes = classes
        self.features = features
        self.score_func = score_func
        self.use_label_for_attr = use_label_for_attr
        self.baseline_cache: dict = {}
        self.transformed_input_cache: dict = {}

    def calculate_predicted_scores(
        self, inputs, additional_forward_args, model
    ) -> Tuple[
        List[OutputScore], Optional[List[Tuple[Tensor, ...]]], Tuple[Tensor, ...]
    ]:
        # Check if inputs have cached baselines and transformed inputs
        hashable_inputs = tuple(inputs)
        if hashable_inputs in self.baseline_cache:
            baselines_group = self.baseline_cache[hashable_inputs]
            transformed_inputs = self.transformed_input_cache[hashable_inputs]
        else:
            # Initialize baselines
            baseline_transforms_len = 1  # todo support multiple baselines
            baselines: List[List[Optional[Tensor]]] = [
                [None] * len(self.features) for _ in range(baseline_transforms_len)
            ]
            transformed_inputs = list(inputs)
            for feature_i, feature in enumerate(self.features):
                transformed_inputs[feature_i] = self._transform(
                    feature.input_transforms, transformed_inputs[feature_i], True
                )
                for baseline_i in range(baseline_transforms_len):
                    if baseline_i > len(feature.baseline_transforms) - 1:
                        baselines[baseline_i][feature_i] = torch.zeros_like(
                            transformed_inputs[feature_i]
                        )
                    else:
                        baselines[baseline_i][feature_i] = self._transform(
                            [feature.baseline_transforms[baseline_i]],
                            transformed_inputs[feature_i],
                            True,
                        )

            baselines = cast(List[List[Optional[Tensor]]], baselines)
            baselines_group = [tuple(b) for b in baselines]
            self.baseline_cache[hashable_inputs] = baselines_group
            self.transformed_input_cache[hashable_inputs] = transformed_inputs

        outputs = _run_forward(
            model,
            tuple(transformed_inputs),
            additional_forward_args=additional_forward_args,
        )

        if self.score_func is not None:
            outputs = self.score_func(outputs)

        if outputs.nelement() == 1:
            scores = outputs
            predicted = scores.round().to(torch.int)
        else:
            scores, predicted = outputs.topk(min(4, outputs.shape[-1]))

        scores = scores.cpu().squeeze(0)
        predicted = predicted.cpu().squeeze(0)

        predicted_scores = self._get_labels_from_scores(scores, predicted)

        return predicted_scores, baselines_group, tuple(transformed_inputs)

    def calculate_attribution(
        self,
        baselines: Optional[Sequence[Tuple[Tensor, ...]]],
        data: Tuple[Tensor, ...],
        additional_forward_args: Optional[Tuple[Tensor, ...]],
        label: Optional[Union[Tensor]],
        attribution_method_name: str,
        attribution_arguments: Dict,
        model: Module,
    ) -> Tuple[Tensor, ...]:
        attribution_cls = ATTRIBUTION_NAMES_TO_METHODS[attribution_method_name]
        attribution_method = attribution_cls(model)
        if attribution_method_name in ATTRIBUTION_METHOD_CONFIG:
            param_config = ATTRIBUTION_METHOD_CONFIG[attribution_method_name]
            if param_config.post_process:
                for k, v in attribution_arguments.items():
                    if k in param_config.post_process:
                        attribution_arguments[k] = param_config.post_process[k](v)

        # TODO support multiple baselines
        baseline = baselines[0] if baselines and len(baselines) > 0 else None
        label = (
            None
            if not self.use_label_for_attr or label is None or label.nelement() == 0
            else label
        )
        if "baselines" in inspect.signature(attribution_method.attribute).parameters:
            attribution_arguments["baselines"] = baseline
        attr = attribution_method.attribute.__wrapped__(
            attribution_method,  # self
            data,
            additional_forward_args=additional_forward_args,
            target=label,
            **attribution_arguments,
        )

        return attr

    def calculate_net_contrib(
        self, attrs_per_input_feature: Tuple[Tensor, ...]
    ) -> List[float]:
        # get the net contribution per feature (input)
        net_contrib = torch.stack(
            [attrib.flatten().sum() for attrib in attrs_per_input_feature]
        )

        # normalise the contribution, s.t. sum(abs(x_i)) = 1
        norm = torch.norm(net_contrib, p=1)
        # if norm is 0, all net_contrib elements are 0
        net_contrib = safe_div(net_contrib, norm)

        return net_contrib.tolist()

    def _transform(
        self, transforms: Iterable[Callable], inputs: Tensor, batch: bool = False
    ) -> Tensor:
        transformed_inputs = inputs
        # TODO support batch size > 1
        if batch:
            transformed_inputs = inputs.squeeze(0)

        for t in transforms:
            transformed_inputs = t(transformed_inputs)

        if batch:
            transformed_inputs = transformed_inputs.unsqueeze(0)

        return transformed_inputs

    def _get_labels_from_scores(
        self, scores: Tensor, indices: Tensor
    ) -> List[OutputScore]:
        pred_scores: List[OutputScore] = []
        if indices.nelement() < 2:
            return pred_scores
        for i in range(len(indices)):
            score = scores[i]
            pred_scores.append(
                OutputScore(score, indices[i], self.classes[int(indices[i])])
            )
        return pred_scores
