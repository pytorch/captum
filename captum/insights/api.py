#!/usr/bin/env python3
from collections import namedtuple
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
from captum.attr import IntegratedGradients
from captum.attr._utils.batching import _batched_generator
from captum.attr._utils.common import _run_forward, safe_div
from captum.insights.features import BaseFeature
from torch import Tensor
from torch.nn import Module


OutputScore = namedtuple("OutputScore", "score index label")
VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted active_index"
)
Contribution = namedtuple("Contribution", "name percent")
SampleCache = namedtuple("SampleCache", "inputs additional_forward_args label")


class FilterConfig(NamedTuple):
    steps: int = 20
    prediction: str = "all"
    classes: List[str] = []
    count: int = 4


class Batch:
    def __init__(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        labels: Optional[Tensor],
        additional_args=None,
    ):
        self.inputs = inputs
        self.labels = labels
        self.additional_args = additional_args


class AttributionVisualizer(object):
    def __init__(
        self,
        models: Union[List[Module], Module],
        classes: List[str],
        features: Union[List[BaseFeature], BaseFeature],
        dataset: Iterable[Batch],
        score_func: Optional[Callable] = None,
        use_label_for_attr: bool = True,
    ):
        if not isinstance(models, List):
            models = [models]

        if not isinstance(features, List):
            features = [features]

        self.models = models
        self.classes = classes
        self.features = features
        self.dataset = dataset
        self.score_func = score_func
        self._outputs = []
        self._config = FilterConfig(steps=25, prediction="all", classes=[], count=4)
        self._use_label_for_attr = use_label_for_attr

    def _calculate_attribution_from_cache(
        self, index: int, target: Optional[Tensor]
    ) -> VisualizationOutput:
        c = self._outputs[index][1]
        return self._calculate_vis_output(
            c.inputs, c.additional_forward_args, c.label, torch.tensor(target)
        )

    def _calculate_attribution(
        self,
        net: Module,
        baselines: Optional[List[Tuple[Tensor, ...]]],
        data: Tuple[Tensor, ...],
        additional_forward_args: Optional[Tuple[Tensor, ...]],
        label: Optional[Union[Tensor]],
    ) -> Tensor:
        ig = IntegratedGradients(net)
        # TODO support multiple baselines
        baseline = baselines[0] if len(baselines) > 0 else None
        label = (
            None
            if not self._use_label_for_attr or label is None or label.nelement() == 0
            else label
        )
        attr_ig = ig.attribute(
            data,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            target=label,
            n_steps=self._config.steps,
        )

        return attr_ig

    def _update_config(self, settings):
        self._config = FilterConfig(
            steps=int(settings["approximation_steps"]),
            prediction=settings["prediction"],
            classes=settings["classes"],
            count=4,
        )

    def render(self):
        from IPython.display import display
        from captum.insights.widget import CaptumInsights

        widget = CaptumInsights(visualizer=self)
        display(widget)

    def serve(self, blocking=False, debug=False):
        from captum.insights.server import start_server

        start_server(self, blocking=blocking, debug=debug)

    def _get_labels_from_scores(
        self, scores: Tensor, indices: Tensor
    ) -> List[OutputScore]:
        pred_scores = []
        for i in range(len(indices)):
            score = scores[i]
            pred_scores.append(OutputScore(score, indices[i], self.classes[indices[i]]))
        return pred_scores

    def _transform(
        self,
        transforms: Union[Callable, List[Callable]],
        inputs: Tensor,
        batch: bool = False,
    ) -> Tensor:
        transformed_inputs = inputs
        # TODO support batch size > 1
        if batch:
            transformed_inputs = inputs.squeeze()

        if isinstance(transforms, List):
            for t in transforms:
                transformed_inputs = t(transformed_inputs)
        else:
            transformed_inputs = transforms(transformed_inputs)

        if batch:
            transformed_inputs = transformed_inputs.unsqueeze(0)

        return transformed_inputs

    def _calculate_net_contrib(self, attrs_per_input_feature: List[Tensor]):
        # get the net contribution per feature (input)
        net_contrib = torch.stack(
            [attrib.flatten().sum() for attrib in attrs_per_input_feature]
        )

        # normalise the contribution, s.t. sum(abs(x_i)) = 1
        norm = torch.norm(net_contrib, p=1)
        net_contrib = safe_div(net_contrib, norm, default_value=net_contrib)

        return net_contrib.tolist()

    def _predictions_matches_labels(
        self, predicted_scores: List[OutputScore], labels: Union[str, List[str]]
    ) -> bool:
        if len(predicted_scores) == 0:
            return False

        predicted_label = predicted_scores[0].label

        if isinstance(labels, List):
            return predicted_label in labels

        return labels == predicted_label

    def _should_keep_prediction(
        self, predicted_scores: List[OutputScore], actual_label: OutputScore
    ) -> bool:
        # filter by class
        if len(self._config.classes) != 0:
            if not self._predictions_matches_labels(
                predicted_scores, self._config.classes
            ):
                return False

        # filter by accuracy
        label_name = actual_label.label
        if self._config.prediction == "all":
            pass
        elif self._config.prediction == "correct":
            if not self._predictions_matches_labels(predicted_scores, label_name):
                return False
        elif self._config.prediction == "incorrect":
            if self._predictions_matches_labels(predicted_scores, label_name):
                return False
        else:
            raise Exception(f"Invalid prediction config: {self._config.prediction}")

        return True

    def _calculate_vis_output(
        self, inputs, additional_forward_args, label, target=None
    ) -> Optional[VisualizationOutput]:
        net = self.models[0]  # TODO process multiple models

        # initialize baselines
        baseline_transforms_len = len(self.features[0].baseline_transforms or [])
        baselines = [
            [None] * len(self.features) for _ in range(baseline_transforms_len)
        ]
        transformed_inputs = list(inputs)

        # transformed_inputs = list([i.clone() for i in inputs])
        for feature_i, feature in enumerate(self.features):
            if feature.input_transforms is not None:
                transformed_inputs[feature_i] = self._transform(
                    feature.input_transforms, transformed_inputs[feature_i], True
                )
            if feature.baseline_transforms is not None:
                assert baseline_transforms_len == len(
                    feature.baseline_transforms
                ), "Must have same number of baselines across all features"

                for baseline_i, baseline_transform in enumerate(
                    feature.baseline_transforms
                ):
                    baselines[baseline_i][feature_i] = self._transform(
                        baseline_transform, transformed_inputs[feature_i], True
                    )

        outputs = _run_forward(
            net,
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

        if label is not None and len(label) > 0:
            actual_label_output = OutputScore(
                score=100, index=label[0], label=self.classes[label[0]]
            )
        else:
            actual_label_output = None

        predicted_scores = self._get_labels_from_scores(scores, predicted)

        # Filter based on UI configuration
        if not self._should_keep_prediction(predicted_scores, actual_label_output):
            return None

        baselines = [tuple(b) for b in baselines]

        if target is None:
            target = predicted_scores[0].index if len(predicted_scores) > 0 else None

        # attributions are given per input*
        # inputs given to the model are described via `self.features`
        #
        # *an input contains multiple features that represent it
        #   e.g. all the pixels that describe an image is an input

        attrs_per_input_feature = self._calculate_attribution(
            net, baselines, tuple(transformed_inputs), additional_forward_args, target
        )

        net_contrib = self._calculate_net_contrib(attrs_per_input_feature)

        # the features per input given
        features_per_input = [
            feature.visualize(attr, data, contrib)
            for feature, attr, data, contrib in zip(
                self.features, attrs_per_input_feature, inputs, net_contrib
            )
        ]

        return VisualizationOutput(
            feature_outputs=features_per_input,
            actual=actual_label_output,
            predicted=predicted_scores,
            active_index=target if target is not None else actual_label_output.index,
        )

    def _get_outputs(self) -> List[VisualizationOutput]:
        batch_data = next(self.dataset)
        vis_outputs = []

        for inputs, additional_forward_args, label in _batched_generator(
            inputs=batch_data.inputs,
            additional_forward_args=batch_data.additional_args,
            target_ind=batch_data.labels,
            internal_batch_size=1,  # should be 1 until we have batch label support
        ):
            output = self._calculate_vis_output(inputs, additional_forward_args, label)
            if output is not None:
                cache = SampleCache(inputs, additional_forward_args, label)
                vis_outputs.append((output, cache))

        return vis_outputs

    def visualize(self):
        self._outputs = []
        while len(self._outputs) < self._config.count:
            try:
                self._outputs.extend(self._get_outputs())
            except StopIteration:
                break
        return [o[0] for o in self._outputs]
