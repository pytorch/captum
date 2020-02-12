#!/usr/bin/env python3
import inspect
from collections import namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
from captum.attr import IntegratedGradients
from captum.attr._utils.batching import _batched_generator
from captum.attr._utils.common import _run_forward, safe_div
from captum.insights.config import (
    ATTRIBUTION_METHOD_CONFIG,
    ATTRIBUTION_NAMES_TO_METHODS,
)
from captum.insights.features import BaseFeature
from captum.insights.server import namedtuple_to_dict
from torch import Tensor
from torch.nn import Module


_CONTEXT_COLAB = "_CONTEXT_COLAB"
_CONTEXT_IPYTHON = "_CONTEXT_IPYTHON"
_CONTEXT_NONE = "_CONTEXT_NONE"


def _get_context():
    """Determine the most specific context that we're in.
    Implementation from TensorBoard: https://git.io/JvObD.

    Returns:
    _CONTEXT_COLAB: If in Colab with an IPython notebook context.
    _CONTEXT_IPYTHON: If not in Colab, but we are in an IPython notebook
      context (e.g., from running `jupyter notebook` at the command
      line).
    _CONTEXT_NONE: Otherwise (e.g., by running a Python script at the
      command-line or using the `ipython` interactive shell).
    """
    # In Colab, the `google.colab` module is available, but the shell
    # returned by `IPython.get_ipython` does not have a `get_trait`
    # method.
    try:
        import google.colab  # noqa: F401
        import IPython
    except ImportError:
        pass
    else:
        if IPython.get_ipython() is not None:
            # We'll assume that we're in a Colab notebook context.
            return _CONTEXT_COLAB

    # In an IPython command line shell or Jupyter notebook, we can
    # directly query whether we're in a notebook context.
    try:
        import IPython
    except ImportError:
        pass
    else:
        ipython = IPython.get_ipython()
        if ipython is not None and ipython.has_trait("kernel"):
            return _CONTEXT_IPYTHON

    # Otherwise, we're not in a known notebook context.
    return _CONTEXT_NONE


OutputScore = namedtuple("OutputScore", "score index label")
VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted active_index"
)
Contribution = namedtuple("Contribution", "name percent")
SampleCache = namedtuple("SampleCache", "inputs additional_forward_args label")


class FilterConfig(NamedTuple):
    attribution_method: str = IntegratedGradients.get_name()
    attribution_arguments: Dict[str, Any] = {
        arg: config.value
        for arg, config in ATTRIBUTION_METHOD_CONFIG[
            IntegratedGradients.get_name()
        ].items()
    }
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
        r"""
        Constructs batch of inputs to be attributed and visualized.

        Args:

            inputs (tensor or tuple of tensors): Batch of inputs for a model.
                        These may be either a Tensor or tuple of tensors. Each tensor
                        must correspond to a feature for AttributionVisualizer, and
                        the corresponding input transform function of the feature
                        is applied to each input tensor prior to passing it to the
                        model. It is assumed that the first dimension of each
                        input tensor corresponds to the number of examples
                        (batch size) and is aligned for all input tensors.
            labels (tensor): Tensor containing correct labels for input examples.
                        This must be a 1D tensor with length matching the first
                        dimension of each input tensor.
            additional_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to ``forward_func`` in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples.
        """
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
        r"""
        Args:

            models (torch.nn.module): PyTorch module (model) for attribution
                          visualization.
                          We plan to support visualizing and comparing multiple models
                          in the future, but currently this supports only a single
                          model.
            classes (list of string): List of strings corresponding to the names of
                          classes for classification.
            features (list of BaseFeature): List of BaseFeatures, which correspond
                          to input arguments to the model. Each feature object defines
                          relevant transformations for converting to model input,
                          constructing baselines, and visualizing. The length of the
                          features list should exactly match the number of (tensor)
                          arguments expected by the given model.
                          For instance, an image classifier should only provide
                          a single BaseFeature, while a multimodal classifier may
                          provide a list of features, each corresponding to a different
                          tensor input and potentially different modalities.
            dataset (iterable of Batch): Defines the dataset to visualize attributions
                          for. This must be an iterable of batch objects, each of which
                          may contain multiple input examples.
            score_func (callable, optional): This function is applied to the model
                          output to obtain the score for each class. For instance,
                          this function could be the softmax or final non-linearity
                          of the network, applied to the model output. The indices
                          of the second dimension of the output should correspond
                          to the class names provided. If None, the model outputs
                          are taken directly and assumed to correspond to the
                          class scores.
                          Default: None
            use_label_for_attr (boolean, optional): If true, the class index is passed
                          to the relevant attribution method. This is necessary in most
                          cases where there is an output neuron corresponding to each
                          class. When the model output is a scalar and class index
                          (e.g. positive, negative) is inferred from the output value,
                          this argument should be False.
                          Default: True
        """
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
        self._config = FilterConfig(prediction="all", classes=[], count=4)
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
        attribution_cls = ATTRIBUTION_NAMES_TO_METHODS[self._config.attribution_method]
        attribution_method = attribution_cls(net)
        args = self._config.attribution_arguments
        # TODO support multiple baselines
        baseline = baselines[0] if len(baselines) > 0 else None
        label = (
            None
            if not self._use_label_for_attr or label is None or label.nelement() == 0
            else label
        )
        if "baselines" in inspect.signature(attribution_method.attribute).parameters:
            args["baselines"] = baseline
        attr = attribution_method.attribute(
            data, additional_forward_args=additional_forward_args, target=label, **args
        )

        return attr

    def _update_config(self, settings):
        self._config = FilterConfig(
            attribution_method=settings["attribution_method"],
            attribution_arguments=settings["arguments"],
            prediction=settings["prediction"],
            classes=settings["classes"],
            count=4,
        )

    def render(self):
        from IPython.display import display
        from captum.insights.widget import CaptumInsights

        widget = CaptumInsights(visualizer=self)
        display(widget)

    def serve(self, blocking=False, debug=False, port=None):
        context = _get_context()
        if context == _CONTEXT_COLAB:
            self._serve_colab(blocking=blocking, debug=debug, port=port)
        else:
            self._serve(blocking=blocking, debug=debug, port=port)

    def _serve(self, blocking=False, debug=False, port=None):
        from captum.insights.server import start_server

        start_server(self, blocking=blocking, debug=debug, _port=port)

    def _serve_colab(self, blocking=False, debug=False, port=None):
        from IPython.display import display, HTML
        from captum.insights.server import start_server
        import ipywidgets as widgets

        # TODO: Output widget only captures beginning of server logs. It seems
        # the context manager isn't respected when the web server is run on a
        # separate thread. We should fix to display entirety of the logs
        out = widgets.Output()
        with out:
            port = start_server(self, blocking=blocking, debug=debug, _port=port)
        shell = """
            <div id="root"></div>
            <script>
            (function() {
              document.querySelector("base").href = "http://localhost:%PORT%";
              function reloadScriptsAndCSS(root) {
                // Referencing TensorBoard's method for reloading scripts,
                // we remove and reinsert each script
                for (const script of root.querySelectorAll("script")) {
                  const newScript = document.createElement("script");
                  newScript.type = script.type;
                  if (script.src) {
                    newScript.src = script.src;
                  }
                  if (script.textContent) {
                    newScript.textContent = script.textContent;
                  }
                  root.appendChild(newScript);
                  script.remove();
                }
                // A similar method is used to reload styles
                for (const link of root.querySelectorAll("link")) {
                  const newLink = document.createElement("link");
                  newLink.rel = link.rel;
                  newLink.href = link.href;
                  document.querySelector("head").appendChild(newLink);
                  link.remove();
                }
              }
              const root = document.getElementById("root");
              fetch(".")
                .then(x => x.text())
                .then(html => void (root.innerHTML = html))
                .then(() => reloadScriptsAndCSS(root));
            })();
            </script>
        """.replace(
            "%PORT%", str(port)
        )
        html = HTML(shell)
        display(html)
        display(out)

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

    def get_insights_config(self):
        return {
            "classes": self.classes,
            "methods": list(ATTRIBUTION_NAMES_TO_METHODS.keys()),
            "method_arguments": namedtuple_to_dict(ATTRIBUTION_METHOD_CONFIG),
            "selected_method": self._config.attribution_method,
        }
