#!/usr/bin/env python3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import Tensor

from captum.attr._utils.common import _format_tensor_into_tuples
from captum.attr._utils.stat import Stat
from captum.attr._utils.summarizer import Summarizer


class ClassSummarizer(Summarizer):
    r"""
    Used to keep track of summaries for associated classes. The
    classes/labels can be of any type that are supported by `dict`.

    This also keeps track of an aggregate of all class summaries.
    """

    def __init__(self, stats: List[Stat]):
        Summarizer.__init__(self, stats)
        self.summaries: Dict[Any, Summarizer] = defaultdict(
            lambda: Summarizer(stats=stats)
        )

    def update(
        self,
        x: Union[Tensor, Tuple[Tensor, ...]],
        labels: Optional[Union[Any, List[Any]]] = None,
    ):
        r"""
        Updates the stats of the summarizer, optionally associated to classes.

        This accepts either a single tensor to summarise or a tuple of tensors.

        Args:
            x (Tensor or Tuple[Tensor, ...]):
                The input tensor to be summarised. The first
                dimension of this input must be associated to
                the batch size of the inputs.
            labels (Any, List[Any], optional):
                The associated labels for `x`. If Any, we
                assume `labels` represents the label for all inputs in `x`.

                If this is None we simply aggregate the total summary.
        """
        if labels is None:
            super().update(x)
            return

        x = _format_tensor_into_tuples(x)

        if not isinstance(labels, list):
            labels = [labels]

        for tensor in x:
            assert tensor.size(0) == len(labels)

        for i, label in enumerate(labels):
            tensors_to_summarize = tuple(tensor[i] for tensor in x)

            self.summaries[label].update(tensors_to_summarize)
            super().update(tensors_to_summarize)

    @property
    def class_summaries(self) -> Dict[Any, Dict[str, Tensor]]:
        r"""
        Returns:
             The summaries for each class a dictionary.
        """
        return {key: value.summary for key, value in self.summaries.items()}
