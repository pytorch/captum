#!/usr/bin/env python3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from captum.attr._utils.common import _format_tensor_into_tuples
from captum.attr._utils.stat import Stat
from captum.attr._utils.summarizer import Summarizer


class ClassSummarizer(Summarizer):
    """
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
        """
        Updates the stats of the summarizer, optionally associated to classes.

        This accepts either a single tensor to summarise or a batch of tensors.
        If it is a batch of tensors, then `labels` must be a list.

        Args:
            x (Tensor or Tuple[Tensor, ...]):
                The input tensor to be summarised. The first
                dimension of this input must be associated to
                the amount of `labels` specified; unless there
                is one labels represents one label (in that case
                the first dimension can optionally be unsqueezed).
            labels (Any, List[Any], optional):
                The associated labels for `x`. If Any, we
                assume x does not represent a batch of inputs.

                If this is None we simply aggregate the total summary.
        """
        if labels is None:
            super().update(x)
            return

        x = _format_tensor_into_tuples(x)

        should_resqueeze = torch.zeros(len(x))

        batched_x = x
        if not isinstance(labels, list):
            # for single input we need to ensure
            # the input is (1, ...) in order to index it
            # thus we do-so with a squeeze and after we're
            # done we need to re-squeeze the input to
            # where it was such that it seems like we did not
            # touch the given tensor
            labels = [labels]
            for i in range(len(x)):
                should_resqueeze[i] = len(x[i].size()) == 0 or x[i].size(0) != 1
            batched_x = tuple(y.unsqueeze(0) for y in x)

        for i, label in enumerate(labels):
            tensors_to_summarize = tuple(
                tensor[i].squeeze(0) if resqueeze else tensor[i]
                for tensor, resqueeze in zip(batched_x, should_resqueeze)
            )

            self.summaries[label].update(tensors_to_summarize)
            super().update(tensors_to_summarize)

    @property
    def class_summaries(self) -> Dict[Any, Dict[str, Tensor]]:
        """
        Returns:
             The summaries for each class a dictionary.
        """
        return {key: value.summary for key, value in self.summaries.items()}
