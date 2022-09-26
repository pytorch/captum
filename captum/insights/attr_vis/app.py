#!/usr/bin/env python3
from collections import namedtuple
from itertools import cycle
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
from captum.insights.attr_vis.attribution_calculation import (
    AttributionCalculation,
    OutputScore,
)
from captum.insights.attr_vis.config import (
    ATTRIBUTION_METHOD_CONFIG,
    ATTRIBUTION_NAMES_TO_METHODS,
)
from captum.insights.attr_vis.features import BaseFeature
from captum.insights.attr_vis.server import namedtuple_to_dict
from captum.log import log_usage
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


VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_outputs actual predicted active_index model_index"
)
Contribution = namedtuple("Contribution", "name percent")
SampleCache = namedtuple("SampleCache", "inputs additional_forward_args label")


class FilterConfig(NamedTuple):
    attribution_method: str = IntegratedGradients.get_name()
    # issue with mypy github.com/python/mypy/issues/8376
    attribution_arguments: Dict[str, Any] = {
        arg: config.value  # type: ignore
        for arg, config in ATTRIBUTION_METHOD_CONFIG[
            IntegratedGradients.get_name()
        ].params.items()
    }
    prediction: str = "all"
    classes: List[str] = []
    num_examples: int = 4


class Batch:
    def __init__(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        labels: Optional[Tensor],
        additional_args=None,
    ) -> None:
        r"""
        Constructs batch of inputs to be attributed and visualized.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Batch of inputs for a model.
                        These may be either a Tensor or tuple of tensors. Each tensor
                        must correspond to a feature for AttributionVisualizer, and
                        the corresponding input transform function of the feature
                        is applied to each input tensor prior to passing it to the
                        model. It is assumed that the first dimension of each
                        input tensor corresponds to the number of examples
                        (batch size) and is aligned for all input tensors.
            labels (Tensor): Tensor containing correct labels for input examples.
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


class AttributionVisualizer:
    def __init__(
        self,
        models: Union[List[Module], Module],
        classes: List[str],
        features: Union[List[BaseFeature], BaseFeature],
        dataset: Iterable[Batch],
        score_func: Optional[Callable] = None,
        use_label_for_attr: bool = True,
    ) -> None:
        r"""
        Args:

            models (torch.nn.Module): One or more PyTorch modules (models) for
                          attribution visualization.
            classes (list[str]): List of strings corresponding to the names of
                          classes for classification.
            features (list[BaseFeature]): List of BaseFeatures, which correspond
                          to input arguments to the model. Each feature object defines
                          relevant transformations for converting to model input,
                          constructing baselines, and visualizing. The length of the
                          features list should exactly match the number of (tensor)
                          arguments expected by the given model.
                          For instance, an image classifier should only provide
                          a single BaseFeature, while a multimodal classifier may
                          provide a list of features, each corresponding to a different
                          tensor input and potentially different modalities.
            dataset (Iterable of Batch): Defines the dataset to visualize attributions
                          for. This must be an iterable of batch objects, each of which
                          may contain multiple input examples.
            score_func (Callable, optional): This function is applied to the model
                          output to obtain the score for each class. For instance,
                          this function could be the softmax or final non-linearity
                          of the network, applied to the model output. The indices
                          of the second dimension of the output should correspond
                          to the class names provided. If None, the model outputs
                          are taken directly and assumed to correspond to the
                          class scores.
                          Default: None
            use_label_for_attr (bool, optional): If true, the class index is passed
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

        self.classes = classes
        self.features = features
        self.dataset = dataset
        self.models = models
        self.attribution_calculation = AttributionCalculation(
            models, classes, features, score_func, use_label_for_attr
        )
        self._outputs: List[VisualizationOutput] = []
        self._config = FilterConfig(prediction="all", classes=[], num_examples=4)
        self._dataset_iter = iter(dataset)
        self._dataset_cache: List[Batch] = []

    def _calculate_attribution_from_cache(
        self, input_index: int, model_index: int, target: Optional[Tensor]
    ) -> Optional[VisualizationOutput]:
        c = self._outputs[input_index][1]
        result = self._calculate_vis_output(
            c.inputs,
            c.additional_forward_args,
            c.label,
            torch.tensor(target),
            model_index,
        )

        if not result:
            return None
        return result[0]

    def _update_config(self, settings):
        self._config = FilterConfig(
            attribution_method=settings["attribution_method"],
            attribution_arguments=settings["arguments"],
            prediction=settings["prediction"],
            classes=settings["classes"],
            num_examples=4,
        )

    @log_usage()
    def render(self, debug=True):
        from captum.insights.attr_vis.widget import CaptumInsights
        from IPython.display import display

        widget = CaptumInsights(visualizer=self)
        display(widget)
        if debug:
            display(widget.out)

    @log_usage()
    def serve(self, blocking=False, debug=False, port=None, bind_all=False):
        context = _get_context()
        if context == _CONTEXT_COLAB:
            return self._serve_colab(blocking=blocking, debug=debug, port=port)
        else:
            return self._serve(
                blocking=blocking, debug=debug, port=port, bind_all=bind_all
            )

    def _serve(self, blocking=False, debug=False, port=None, bind_all=False):
        from captum.insights.attr_vis.server import start_server

        return start_server(
            self, blocking=blocking, debug=debug, _port=port, bind_all=bind_all
        )

    def _serve_colab(self, blocking=False, debug=False, port=None):
        import ipywidgets as widgets
        from captum.insights.attr_vis.server import start_server
        from IPython.display import display, HTML

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
        self, predicted_scores: List[OutputScore], actual_label: Optional[OutputScore]
    ) -> bool:
        # filter by class
        if len(self._config.classes) != 0:
            if not self._predictions_matches_labels(
                predicted_scores, self._config.classes
            ):
                return False

        if not actual_label:
            return True

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
        self,
        inputs,
        additional_forward_args,
        label,
        target=None,
        single_model_index=None,
    ) -> Optional[List[VisualizationOutput]]:
        # Use all models, unless the user wants to render data for a particular one
        models_used = (
            [self.models[single_model_index]]
            if single_model_index is not None
            else self.models
        )
        results = []
        for model_index, model in enumerate(models_used):
            # Get list of model visualizations for each input
            actual_label_output = None
            if label is not None and len(label) > 0:
                label_index = int(label[0])
                actual_label_output = OutputScore(
                    score=100, index=label_index, label=self.classes[label_index]
                )

            (
                predicted_scores,
                baselines,
                transformed_inputs,
            ) = self.attribution_calculation.calculate_predicted_scores(
                inputs, additional_forward_args, model
            )

            # Filter based on UI configuration
            if actual_label_output is None or not self._should_keep_prediction(
                predicted_scores, actual_label_output
            ):
                continue

            if target is None:
                target = (
                    predicted_scores[0].index if len(predicted_scores) > 0 else None
                )

            # attributions are given per input*
            # inputs given to the model are described via `self.features`
            #
            # *an input contains multiple features that represent it
            #   e.g. all the pixels that describe an image is an input

            attrs_per_feature = self.attribution_calculation.calculate_attribution(
                baselines,
                transformed_inputs,
                additional_forward_args,
                target,
                self._config.attribution_method,
                self._config.attribution_arguments,
                model,
            )

            net_contrib = self.attribution_calculation.calculate_net_contrib(
                attrs_per_feature
            )

            # the features per input given
            features_per_input = [
                feature.visualize(attr, data, contrib)
                for feature, attr, data, contrib in zip(
                    self.features, attrs_per_feature, inputs, net_contrib
                )
            ]

            results.append(
                VisualizationOutput(
                    feature_outputs=features_per_input,
                    actual=actual_label_output,
                    predicted=predicted_scores,
                    active_index=target
                    if target is not None
                    else actual_label_output.index,
                    # Even if we only iterated over one model, the index should be fixed
                    # to show the index the model would have had in the list
                    model_index=single_model_index
                    if single_model_index is not None
                    else model_index,
                )
            )

        return results if results else None

    def _get_outputs(self) -> List[Tuple[List[VisualizationOutput], SampleCache]]:
        # If we run out of new batches, then we need to
        # display data which was already shown before.
        # However, since the dataset given to us is a generator,
        # we can't reset it to return to the beginning.
        # Because of this, we store a small cache of stale
        # data, and iterate on it after the main generator
        # stops returning new batches.
        try:
            batch_data = next(self._dataset_iter)
            self._dataset_cache.append(batch_data)
            if len(self._dataset_cache) > self._config.num_examples:
                self._dataset_cache.pop(0)
        except StopIteration:
            self._dataset_iter = cycle(self._dataset_cache)
            batch_data = next(self._dataset_iter)

        vis_outputs = []

        # Type ignore for issue with passing union to function taking generic
        # https://github.com/python/mypy/issues/1533
        for (
            inputs,
            additional_forward_args,
            label,
        ) in _batched_generator(  # type: ignore
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

    @log_usage()
    def visualize(self):
        self._outputs = []
        while len(self._outputs) < self._config.num_examples:
            self._outputs.extend(self._get_outputs())
        return [o[0] for o in self._outputs]

    def get_insights_config(self):
        return {
            "classes": self.classes,
            "methods": list(ATTRIBUTION_NAMES_TO_METHODS.keys()),
            "method_arguments": namedtuple_to_dict(
                {k: v.params for (k, v) in ATTRIBUTION_METHOD_CONFIG.items()}
            ),
            "selected_method": self._config.attribution_method,
        }
