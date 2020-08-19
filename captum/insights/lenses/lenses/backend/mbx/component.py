#!/usr/bin/env python3
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from mypy_extensions import TypedDict
import numpy as np
from collections import namedtuple


if TYPE_CHECKING:
    from .module import Module
    from .explorer import Explorer

from .module import ModuleFullId


class ComponentFullId:
    def __init__(self, id: int, module: ModuleFullId):
        self.id: int = id
        self.module: ModuleFullId = module

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "module": self.module.to_dict()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ComponentFullId":
        return ComponentFullId(data["id"], ModuleFullId.from_dict(data["module"]))


Sample = namedtuple("Sample", ["id", "value"])


class SampleBin:
    r"""
        A bin of scalar samples whose values are
        greater than or equal to 'begin' and
        less than 'end'.
    """

    def __init__(self, begin: Optional[float], end: Optional[float]):
        self.begin: Optional[float] = begin
        self.end: Optional[float] = end
        self.samples: List[Sample] = []

    def sort_samples(self):
        self.samples.sort(key=lambda sample: sample.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "begin": self.begin,
            "end": self.end,
            "sorted_sample_ids": [sample.id for sample in self.samples],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SampleBin":
        sample_bin = SampleBin(data["begin"], data["end"])
        for sample_id in data["sorted_sample_ids"]:
            # TODO consider storing / retrieving sample values
            sample_bin.samples.append(Sample(sample_id, None))
        return sample_bin


class Component:
    r"""
        The output of a PyTorch module is often high dimensional.
        For easier analysis, we perform dimensionality reduction
        to get a small number of components describing the high-dimensional output.
        This class stores data associated with one component of the output of
        a PyTorch module evaluated on a batch of samples.
    """

    def __init__(
        self,
        module: "Module",
        id: int,
        value: float,
        samples: Optional[np.ndarray],
        compute_sample_bins: bool = True,
        num_sample_bins: int = 10,
    ):
        if samples is not None:
            assert (
                len(samples.shape) == 1
            ), f"expected samples as 1D array (1 component per sample), got {samples.shape}"

        self.module = module
        self.id = id
        self.value = value
        self.samples = samples

        if compute_sample_bins:
            if samples is None:
                raise ValueError("samples are required to compute sample bins")
            max_value = samples.max().item()
            min_value = samples.min().item()
            span = max_value - min_value
            boundaries = np.array(
                [
                    min_value + (i + 1) * span / num_sample_bins
                    for i in range(num_sample_bins - 1)
                ]
            )
            bin_ids = np.digitize(samples, boundaries)
            self.sample_bins: List[SampleBin] = []

            begin = None
            for end in boundaries:
                self.sample_bins.append(SampleBin(begin, end))
                begin = end
            self.sample_bins.append(SampleBin(begin, None))
            assert (
                len(self.sample_bins) == num_sample_bins
            ), f"sample_bins must have {num_sample_bins} elements, found {len(self.sample_bins)}"

            for sample_id, bin_id in enumerate(bin_ids):
                sample_value = samples[sample_id].item()
                sample_bin = self.sample_bins[bin_id]
                sample_bin.samples.append(Sample(sample_id, sample_value))

            self.sort_sample_bins()
        else:
            self.sample_bins = []

    def sort_sample_bins(self):
        for sample_bin in self.sample_bins:
            sample_bin.sort_samples()

    def compute_correlation(self, b: "Component"):
        # the correlation between different variables
        # is in the off-diagonal entries ([0, 1] or [1, 0]) of
        # the symmetric 2x2 matrix returned by np.corrcoef
        value = np.corrcoef(self.samples, b.samples)[0, 1].item()
        return Correlation(self, b, value)

    @property
    def full_id(self) -> ComponentFullId:
        return ComponentFullId(self.id, self.module.full_id)

    def to_dict(self, samples: bool = True) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "value": self.value,
            "sample_bins": [sample_bin.to_dict() for sample_bin in self.sample_bins],
        }
        if samples:
            data["samples"] = self.samples.tolist()
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any], module: "Module", id: int) -> "Component":
        samples = np.array(data["samples"], dtype=np.float32)
        component = Component(
            module, id, data["value"], samples, compute_sample_bins=False
        )
        if "sample_bins" in data:
            component.sample_bins = [
                SampleBin.from_dict(sample_bin_data)
                for sample_bin_data in data["sample_bins"]
            ]
        return component


CorrelationData = TypedDict(
    "CorrelationData", {"a": Dict[str, Any], "b": Dict[str, Any], "value": float}
)


class Correlation:
    def __init__(self, a: Component, b: Component, value: float):
        self.a = a
        self.b = b
        self.value = value

    def to_dict(self):
        return {
            "a": self.a.full_id.to_dict(),
            "b": self.b.full_id.to_dict(),
            "value": self.value,
        }

    @staticmethod
    def from_dict(data: CorrelationData, explorer: "Explorer"):
        a_id = ComponentFullId.from_dict(data["a"])
        b_id = ComponentFullId.from_dict(data["b"])
        a = explorer.get_component_by_full_id(a_id)
        b = explorer.get_component_by_full_id(b_id)
        return Correlation(a, b, data["value"])
