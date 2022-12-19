#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any

from torch.nn import Module
from torch.utils.data import Dataset


class DataInfluence(ABC):
    r"""
    An abstract class to define model data influence skeleton.
    """

    def __init_(self, model: Module, train_dataset: Dataset, **kwargs: Any) -> None:
        r"""
        Args:
            model (torch.nn.Module): An instance of pytorch model.
            train_dataset (torch.utils.data.Dataset): PyTorch Dataset that is
                    used to create a PyTorch Dataloader to iterate over the dataset and
                    its labels. This is the dataset for which we will be seeking for
                    influential instances. In most cases this is the training dataset.
            **kwargs: Additional key-value arguments that are necessary for specific
                    implementation of `DataInfluence` abstract class.
        """
        self.model = model
        self.train_dataset = train_dataset

    @abstractmethod
    def influence(self, inputs: Any = None, **kwargs: Any) -> Any:
        r"""
        Args:
            inputs (Any): Batch of examples for which influential
                    instances are computed. They are passed to the forward_func. If
                    `inputs` if a tensor or tuple of tensors, the first dimension
                    of a tensor corresponds to the batch dimension.
            **kwargs: Additional key-value arguments that are necessary for specific
                    implementation of `DataInfluence` abstract class.

        Returns:
            influences (Any): We do not add restrictions on the return type for now,
                    though this may change in the future.
        """
        pass
