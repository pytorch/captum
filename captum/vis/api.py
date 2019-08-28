from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional

from captum import IntegratedGradients

import torch
import torch.nn as nn
from features import BaseFeature
from serve import start_server

ModelScore = namedtuple("ModelScore", "score label")
VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted"
)
Contribution = namedtuple("Contribution", "name percent")


class AttributionVisualizer(object):
    def __init__(
        self,
        models: Any,
        classes: List[str],
        features: List[BaseFeature],
        dataset: Any,
        batch_size: int = 10,
    ):
        self.classes = classes
        self.features = features
        self.models = models
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = iter(
            torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
        )

    def _calculate_attribution(self, net, data, label):
        input = data.unsqueeze(0)
        input.requires_grad = True
        net.eval()
        ig = IntegratedGradients(net)
        net.zero_grad()
        attr_ig, _ = ig.attribute(input, baselines=input * 0, target=label)
        return attr_ig

    def render(self):
        port = start_server(self)
        from IPython.display import IFrame, display

        display(IFrame(src=f"http://127.0.0.1:{port}", width="100%", height="400px"))

    def _get_labels_from_scores(self, scores, indices):
        l = []
        for i in range(len(indices)):
            score = scores[i].item()
            if score > 0.0001:
                l.append(ModelScore(scores[i].item(), self.classes[indices[i]]))
        return l

    def visualize(self):
        data, labels = self.dataloader.next()
        net = self.models[0]  # TODO process multiple models
        outputs = net(data)

        scores, predicted = (
            torch.nn.functional.softmax(outputs, 1).cpu().detach().topk(4)
        )
        outputs = []
        for i in range(self.batch_size):
            datum, label = data[i], labels[i]

            attribution = self._calculate_attribution(net, datum, label)
            for j, feature in enumerate(self.features):
                output = feature.visualize(attribution[j], datum, label)
            predicted_labels = self._get_labels_from_scores(scores[i], predicted[i])
            actual_label = self.classes[labels[i]]
            outputs.append(
                VisualizationOutput(
                    feature_outputs=[output],
                    actual=actual_label,
                    predicted=predicted_labels,
                )
            )
        return outputs
