#!/usr/bin/env python3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from captum.attr._utils.stat import Stat
from captum.attr._utils.summarizer import Summarizer


class ClassSummarizer:
    """
    Used to keep track of summaries for associated classes. The
    classes/labels can be of any type that is supported by `dict`.

    This also keeps track of an aggregate of all class summaries.
    """

    def __init__(self, stats: List[Stat]):
        self.stats = stats
        self.all_summary = Summarizer(stats=stats)
        self.summaries: Dict[Any, Summarizer] = defaultdict(lambda: Summarizer(stats=stats))

    def update(
        self,
        x: Union[Tensor, Tuple[Tensor, ...]],
        labels: Optional[Union[Any, List[Any]]] = None,
    ):
        """
        Updates the stats of the summarizer, optionally associated to classes.

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
        if labels is not None:
            if not isinstance(x, tuple):
                x = (x,)

            should_resqueeze = torch.zeros(len(x))
            if not isinstance(labels, list):
                # we need to support the data having
                # size (1, ...) or (...)
                labels = [labels]
                for i in range(len(x)):
                    should_resqueeze[i] = len(x[i].size()) == 0 or x[i].size(0) != 1
                x = tuple(y.unsqueeze(0) for y in x)

            for i, label in enumerate(labels):
                xx = tuple(
                    y[i].squeeze(0) if resqueeze else y[i]
                    for y, resqueeze in zip(x, should_resqueeze)
                )

                self.summaries[label].update(xx)
                self.all_summary.update(xx)
        else:
            self.all_summary.update(x)

    @property
    def summary(self) -> Dict[str, Tensor]:
        """
        This is equivalent to `Summarizer.summary`.

        Returns:
            An aggregate summary for all classes, represented
            as a dict.
        """
        return self.all_summary.summary

    @property
    def class_summaries(self) -> Dict[Any, Dict[str, Tensor]]:
        """
        Returns:
             The summaries for each class a dictionary.
        """
        return {key: value.summary for key, value in self.summaries.items()}
