# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import random
from typing import Any, Dict, List, Tuple, Union


class ProductBaselines:
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
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        baseline_values: Union[
            List[List[Any]],
            Dict[Union[str, Tuple[str, ...]], List[Any]],
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

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def sample(self) -> Union[List[Any], Dict[str, Any]]:
        baselines = [
            random.choice(baseline_list) for baseline_list in self.baseline_values
        ]

        if not self.dict_keys:
            return baselines

        dict_baselines = {}
        for key, val in zip(self.dict_keys, baselines):
            if not isinstance(key, tuple):
                key, val = (key,), (val,)

            for k, v in zip(key, val):
                dict_baselines[k] = v

        return dict_baselines

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def __call__(self) -> Union[List[Any], Dict[str, Any]]:
        """
        Returns:

            baselines (List or Dict): A sample from the Cartesian product of
                the inputs' available baselines
        """
        return self.sample()
