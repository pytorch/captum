from collections import namedtuple
from typing import Iterable, List, Union

from captum.attr import IntegratedGradients
from captum.insights.features import BaseFeature
from captum.insights.server import start_server

from torch import Module, Tensor

PredictionScore = namedtuple("PredictionScore", "score label")
VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted"
)
Contribution = namedtuple("Contribution", "name percent")


class Transformer(object):
    def __init__(self, transform, name=None):
        self.transform = transform
        self.name = name


class Data(object):
    def __init__(self, inputs, labels, additional_args=None):
        self.inputs = inputs
        self.labels = labels
        self.additional_args = additional_args


class AttributionVisualizer(object):
    def __init__(
        self,
        models: Union[List[Module], Module],
        classes: List[str],
        features: List[BaseFeature],
        dataset: Iterable[Data],
    ):
        self.models = models
        self.classes = classes
        self.features = features
        self.dataset = dataset

    def _calculate_attribution(
        self, net: Module, baselines: List[Tensor], data: Tensor, label: Tensor
    ) -> Tensor:
        # temporary fix until we get full batching support
        data = data.unsqueeze(0)
        for i in range(len(baselines)):
            baselines[i] = baselines[i].unsqueeze(0)

        data.requires_grad = True
        net.eval()
        ig = IntegratedGradients(net)
        net.zero_grad()
        attr_ig, _ = ig.attribute(data, baselines=tuple(baselines), target=label)

        return attr_ig

    def render(self):
        port = start_server(self)
        from IPython.display import IFrame, display

        display(IFrame(src=f"http://127.0.0.1:{port}", width="100%", height="500px"))

    def _get_labels_from_scores(
        self, scores: Tensor, indices: Tensor
    ) -> List[PredictionScore]:
        pred_scores = []
        for i in range(len(indices)):
            score = scores[i].item()
            if score > 0.0001:
                pred_scores.append(
                    PredictionScore(scores[i].item(), self.classes[indices[i]])
                )
        return pred_scores

    def _transform(
        self, transforms: Union[Transformer, List[Transformer]], input: Tensor
    ) -> Tensor:
        if transforms is None:
            return input

        transformed_input = input
        if isinstance(transforms, List):
            for t in transforms:
                transformed_input = t.transform(transformed_input)
        else:
            transformed_input = transforms.transform(transformed_input)

        return transformed_input

    def visualize(self) -> List[VisualizationOutput]:
        batch_data = next(self.dataset)
        net = self.models[0]  # TODO process multiple models
        if batch_data.additional_args is not None:
            outputs = net(batch_data.inputs, batch_data.additional_args)
        else:
            outputs = net(batch_data.inputs)

        scores, predicted = outputs.cpu().detach().topk(4)

        vis_outputs = []
        # convention that batch size is the first index
        for i in range(batch_data.inputs.shape[0]):
            input, label = batch_data.inputs[i], batch_data.labels[i]

            transformed_input = input
            baselines = []

            for feature in self.features:
                baselines.append(self._transform(feature.baseline_transforms, input))
                transformed_input = self._transform(
                    feature.input_transforms, transformed_input
                )

            attribution = self._calculate_attribution(
                net, baselines, transformed_input, label
            )
            for j, feature in enumerate(self.features):
                output = feature.visualize(attribution[j], input, label)

            predicted_labels = self._get_labels_from_scores(scores[i], predicted[i])
            actual_label = self.classes[label]
            vis_outputs.append(
                VisualizationOutput(
                    feature_outputs=[output],
                    actual=actual_label,
                    predicted=predicted_labels,
                )
            )
        return vis_outputs
