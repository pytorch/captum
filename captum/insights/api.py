from collections import namedtuple
from typing import Callable, Iterable, List, Optional, Tuple, Union

from captum.attr import IntegratedGradients
from captum.attr._utils.batching import _batched_generator
from captum.attr._utils.common import _run_forward
from captum.insights.features import BaseFeature

import torch
from torch import Tensor
from torch.nn import Module

PredictionScore = namedtuple("PredictionScore", "score label")
VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted"
)
Contribution = namedtuple("Contribution", "name percent")


class Data:
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
        dataset: Iterable[Data],
        score_func: Optional[Callable] = None,
        n_steps: int = 50,
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
        self.n_steps = n_steps

    def _calculate_attribution(
        self,
        net: Module,
        baselines: List[Tuple[Tensor, ...]],
        data: Tuple[Tensor, ...],
        additional_forward_args: Optional[Tuple[Tensor, ...]],
        label: Optional[Tensor],
    ) -> Tensor:
        ig = IntegratedGradients(net)
        # TODO support multiple baselines
        label = None if label is None or len(label.shape) == 0 else label
        attr_ig, _ = ig.attribute(
            data,
            baselines=baselines[0],
            additional_forward_args=additional_forward_args,
            target=label,
            n_steps=self.n_steps,
        )

        return attr_ig

    def render(self):
        from IPython.display import IFrame, display
        from captum.insights.server import start_server

        port = start_server(self)

        display(IFrame(src=f"http://127.0.0.1:{port}", width="100%", height="500px"))

    def _get_labels_from_scores(
        self, scores: Tensor, indices: Tensor
    ) -> List[PredictionScore]:
        scores, indices = scores.squeeze(), indices.squeeze()
        pred_scores = []
        for i in range(len(indices)):
            score = scores[i].item()
            pred_scores.append(PredictionScore(score, self.classes[indices[i]]))
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
            transformed_inputs.unsqueeze_(0)

        return transformed_inputs

    def _calculate_net_contrib(self, attrs_per_input_feature: List[Tensor]):
        # get the net contribution per feature (input)
        net_contrib = torch.stack(
            [attrib.flatten().sum() for attrib in attrs_per_input_feature]
        )

        # normalise the contribution, s.t. sum(abs(x_i)) = 1
        norm = torch.norm(net_contrib, p=1)
        if norm > 0:
            net_contrib /= norm

        return net_contrib

    def visualize(self) -> List[VisualizationOutput]:
        batch_data = next(self.dataset)
        net = self.models[0]  # TODO process multiple models
        vis_outputs = []

        for i, (inputs, additional_forward_args) in enumerate(
            _batched_generator(
                inputs=batch_data.inputs,
                additional_forward_args=batch_data.additional_args,
                internal_batch_size=1,  # should be 1 until we have batch label support
            )
        ):
            # initialize baselines
            baseline_transforms_len = len(self.features[0].baseline_transforms)
            baselines = [
                [None] * len(self.features) for _ in range(baseline_transforms_len)
            ]
            transformed_inputs = list(inputs)

            for feature_i, feature in enumerate(self.features):
                transformed_inputs[feature_i] = self._transform(
                    feature.input_transforms, transformed_inputs[feature_i], True
                )
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
                net, tuple(transformed_inputs), additional_forward_args
            )

            if self.score_func is not None:
                outputs = self.score_func(outputs)

            label = batch_data.labels[i]

            if len(outputs) == 1:
                scores = outputs
                predicted = scores.round().to(torch.int)
            else:
                scores, predicted = outputs.topk(min(4, len(outputs)))

            scores = scores.cpu().squeeze(0)
            predicted = predicted.cpu().squeeze(0)

            actual_label = self.classes[label]

            predicted_labels = self._get_labels_from_scores(scores, predicted)

            baselines = [tuple(b) for b in baselines]

            # attributions are given per input*
            # inputs given to the model are described via `self.features`
            #
            # *an input contains multiple features that represent it
            #   e.g. all the pixels that describe an image is an input
            attrs_per_input_feature = self._calculate_attribution(
                net,
                baselines,
                tuple(transformed_inputs),
                additional_forward_args,
                label,
            )

            net_contrib = self._calculate_net_contrib(attrs_per_input_feature)

            # the features per input given
            features_per_input = [
                feature.visualize(attr, data, contrib)
                for feature, attr, data, contrib in zip(
                    self.features, attrs_per_input_feature, inputs, net_contrib
                )
            ]

            output = VisualizationOutput(
                feature_outputs=features_per_input,
                actual=actual_label,
                predicted=predicted_labels,
            )

            vis_outputs.append(output)

        return vis_outputs
