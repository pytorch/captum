#!/usr/bin/env python3
import torch
from torch import Tensor
from typing import List, Any, Dict, Union, Optional
from collections import defaultdict
from captum.attr._utils.summarizer import Summarizer
from captum.attr._utils.stat import Stat


class ClassSummarizer:
    """
    TODO
    """

    def __init__(self, stats: List[Stat] = None):
        self.stats = stats
        self.all_summary = Summarizer(stats=stats)
        self.summaries = defaultdict(lambda: Summarizer(stats=stats))

    def update(self, x: Tensor, labels: Optional[Union[Any, List[Any]]] = None):
        """
        Args:
            labels (Any or List[Any])
                If Any, we assume x does not represent a batch of inputs.
        """
        if labels is not None:
            if not isinstance(x, tuple):
                x = (x,)

            # TODO: how to check for list
            should_resqueeze = torch.zeros(len(x))
            if not isinstance(labels, list):
                # we need to support the data having size 1x... or ...
                labels = [labels]
                for i in range(len(x)):
                    should_resqueeze[i] = len(x[i].size()) == 0 or x[i].size(0) != 1
                x = tuple(y.unsqueeze(0) for y in x)

            for i, label in enumerate(labels):
                if should_resqueeze[i]:
                    xx = tuple(y[i].squeeze(0) for y in x)
                else:
                    xx = tuple(y[i] for y in x)

                self.summaries[label].update(xx)
                self.all_summary.update(xx)
        else:
            self.all_summary.update(x)

    @property
    def summary(self) -> Dict[str, Tensor]:
        return self.all_summary.summary

    @property
    def class_summaries(self) -> Dict[Any, Dict[str, Tensor]]:
        return {key: value.summary for key, value in self.summaries.items()}
