from collections import namedtuple
from typing import Iterable, List, Union

from captum.attr import IntegratedGradients
from captum.insights.features import BaseFeature
from captum.insights.server import start_server

from torch import Tensor
from torch.nn import Module

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
        attr_ig, _ = ig.attribute(data, baselines=data * 0, target=3)

        return attr_ig

    def render(self):
        port = start_server(self)
        from IPython.display import IFrame, display

        display(IFrame(src=f"http://127.0.0.1:{port}", width="100%", height="500px"))

    def _get_labels_from_scores(
        self, scores: Tensor, indices: Tensor
    ) -> List[PredictionScore]:
        scores, indices = scores.squeeze(), indices.squeeze()
        pred_scores = []
        for i in range(len(indices)):
            score = scores[i].item()
            if score > 0.0001:
                pred_scores.append(
                    PredictionScore(scores[i].item(), self.classes[indices[i]])
                )
        return pred_scores

    def _transform(
        self,
        transforms: Union[Transformer, List[Transformer]],
        inputs: Tensor,
        batch: bool = False,
    ) -> Tensor:
        if batch:
            for i in range(len(inputs)):
                inputs[i] = self._transform(transforms, inputs[i])
            return inputs

        if transforms is None:
            return inputs

        transformed_inputs = inputs
        if isinstance(transforms, List):
            for t in transforms:
                transformed_inputs = t.transform(transformed_inputs)
        else:
            transformed_inputs = transforms.transform(transformed_inputs)

        return transformed_inputs

    def visualize(self) -> List[VisualizationOutput]:
        batch_data = next(self.dataset)
        net = self.models[0]  # TODO process multiple models

        vis_outputs = []

        # convention that batch size is the first index
        for i in range(batch_data.inputs.shape[0]):
            baselines = []
            transformed_inputs = batch_data.inputs[i]

            for feature in self.features:
                transformed_inputs = self._transform(
                    feature.input_transforms, transformed_inputs
                )
            for feature in self.features:
                baselines.append(
                    self._transform(feature.baseline_transforms, transformed_inputs)
                )

            if batch_data.additional_args is not None:
                outputs = net(
                    transformed_inputs.unsqueeze(0), batch_data.additional_args
                )
            else:
                outputs = net(transformed_inputs.unsqueeze(0))

            scores, predicted = outputs.cpu().detach().topk(4)

            original_input, label = batch_data.inputs[i], batch_data.labels[i]

            attribution = self._calculate_attribution(
                net, baselines, transformed_inputs, label
            )
            for j, feature in enumerate(self.features):
                output = feature.visualize(attribution[j], original_input, label)

            predicted_labels = self._get_labels_from_scores(scores, predicted)
            actual_label = self.classes[label]
            vis_outputs.append(
                VisualizationOutput(
                    feature_outputs=[output],
                    actual=actual_label,
                    predicted=predicted_labels,
                )
            )
        return vis_outputs
