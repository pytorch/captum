# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import random
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Union

GenericBaselineType = TypeVar("GenericBaselineType")


class ProductBaselines(Generic[GenericBaselineType]):
    """
    A Callable Baselines class that returns a sample from the Cartesian product of
    the inputs' available baselines.

    Args:
        baseline_values (List or Dict): A list or dict of lists containing
            the possible values for each feature. If a dict is provided, the keys
            can a string of the feature name and the values is a list of available
            baselines. The keys can also be a tuple of strings to group
            multiple features whose baselines are not independent to each other.
            If the key is a tuple, the value must be a list of tuples of
            the corresponding values.
    """

    def __init__(
        self,
        baseline_values: Union[
            List[List[GenericBaselineType]],
            Dict[Union[str, Tuple[str, ...]], List[GenericBaselineType]],
        ],
    ) -> None:
        if isinstance(baseline_values, dict):
            dict_keys = list(baseline_values.keys())
            baseline_values = [baseline_values[k] for k in dict_keys]
        else:
            dict_keys = []

        # pyre-fixme[4]: Attribute must be annotated.
        self.dict_keys = dict_keys
        self.baseline_values = baseline_values

    def sample(
        self,
    ) -> Union[List[GenericBaselineType], Dict[str, GenericBaselineType]]:
        baselines: List[GenericBaselineType] = [
            random.choice(baseline_list) for baseline_list in self.baseline_values
        ]

        if not self.dict_keys:
            return baselines

        dict_baselines = {}
        for key, val in zip(self.dict_keys, baselines):
            if not isinstance(key, tuple):
                key_tuple, val_tuple = (key,), (val,)
            else:
                key_tuple, val_tuple = key, val

            for k, v in zip(key_tuple, val_tuple):
                dict_baselines[k] = v

        return dict_baselines

    def __call__(
        self,
    ) -> Union[List[GenericBaselineType], Dict[str, GenericBaselineType]]:
        """
        Returns:

            baselines (List or Dict): A sample from the Cartesian product of
                the inputs' available baselines
        """
        return self.sample()
