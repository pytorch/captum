#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
import shutil
import os
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .explorer import Explorer

from .explorer import ExplorerFullId


class ModuleFullId:
    def __init__(self, id: int, explorer: ExplorerFullId):
        self.id: int = id
        self.explorer: ExplorerFullId = explorer

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"id": self.id}
        data["explorer"] = self.explorer.to_dict()
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModuleFullId":
        return ModuleFullId(data["id"], ExplorerFullId.from_dict(data["explorer"]))


from .component import Component


class Module:
    def __init__(
        self, explorer: "Explorer", id: int, name: str, module: Optional[nn.Module]
    ):
        self.explorer = explorer
        self.id = id
        self.name = name
        self.m = module
        self.outputs: List[torch.Tensor] = []
        self.pcs: List[Component] = []
        self.ics: List[Component] = []

    @property
    def full_id(self) -> ModuleFullId:
        return ModuleFullId(self.id, self.explorer.full_id)

    def clear(self):
        self.outputs = []
        self.pcs = []

    def forward_hook(self, module, input, output):
        # TODO handle this in a more general way,
        # for now we assume 2 types of output sizes:
        # (batch_size, features)
        # (batch_size, features, dim1, dim2)
        output_shape_len = len(output.shape)
        if output_shape_len == 2:
            a = output.detach()
        elif len(output.shape) == 4:
            mean_axes = tuple(range(2, output_shape_len))
            a = output.mean(axis=mean_axes).detach()
        else:
            raise ValueError(f"cannot handle output shape {output.shape}")
        assert (
            len(a.shape) == 2
        ), "the shape of the batch output tensor must be (batch_size, num_features)"
        self.outputs.extend(a.to("cpu"))

    def compute_pcs(self, num_components=5):
        self.pcs = []

        num_samples = len(self.outputs)
        assert len(self.outputs[0].shape) == 1, "outputs must be 1D"
        num_features = self.outputs[0].shape[0]
        num_components = min(num_features, num_components)
        if num_components > num_samples:
            raise ValueError("cannot compute more principal components than samples")

        pca = PCA(n_components=num_components, whiten=True)
        projected_samples = pca.fit_transform(torch.stack(self.outputs).numpy())
        assert projected_samples.shape == (
            num_samples,
            num_components,
        ), f"the shape of the projected samples array must be ({num_samples}, {num_components}), got {projected_samples.shape}"

        for i in range(num_components):
            variance_ratio = pca.explained_variance_ratio_[i].item()
            component = Component(self, i, variance_ratio, projected_samples[:, i])
            self.pcs.append(component)

    def compute_ics(self, num_components: int = 5):
        self.ics = []

        num_samples = len(self.outputs)
        assert len(self.outputs[0].shape) == 1, "outputs must be 1D"
        num_features = self.outputs[0].shape[0]
        num_components = min(num_features, num_components)
        if num_components > num_samples:
            raise ValueError("cannot compute more independent components than samples")
        ica = FastICA(n_components=num_components, random_state=0)
        projected_samples = ica.fit_transform(torch.stack(self.outputs).numpy())

        for i in range(num_components):
            component = Component(self, i, 1.0, projected_samples[:, i])
            self.ics.append(component)

    def compute_correlation(
        self, b: "Module", num_components: int, max_iter: int, tol: float
    ) -> "ModuleCorrelation":
        num_features_a: int = self.outputs[0].shape[0]
        num_features_b: int = b.outputs[0].shape[0]
        num_components = min(num_features_a, num_features_b, num_components)
        cca = CCA(n_components=num_components, max_iter=max_iter, tol=tol)
        x = torch.stack(self.outputs).numpy()
        y = torch.stack(b.outputs).numpy()
        cca.fit(x, y)
        y_pred = cca.predict(x)
        score = r2_score(y, y_pred).item()
        return ModuleCorrelation(self, b, score)

    @staticmethod
    def root_filename(dirname: str) -> str:
        return os.path.join(dirname, "module.json")

    @staticmethod
    def outputs_filename(dirname: str) -> str:
        return os.path.join(dirname, "outputs.pt")

    def save(self, dirname: str, overwrite: bool = False, save_outputs: bool = True):
        if os.path.exists(dirname):
            if not overwrite:
                raise RuntimeError(
                    f"{dirname} exists, to overwrite run Module.save({dirname}, overwrite=True)"
                )

            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
            else:
                os.remove(dirname)

        os.makedirs(dirname)
        with open(Module.root_filename(dirname), "w") as f:
            json.dump(self.to_dict(), f)

        if save_outputs:
            self.save_outputs(dirname)

    def save_outputs(self, dirname: str):
        torch.save(self.outputs, Module.outputs_filename(dirname))

    def load_outputs(self, dirname: str):
        outputs_filename = Module.outputs_filename(dirname)
        if not os.path.exists(outputs_filename):
            raise RuntimeError(f"Outputs file does not exist: {outputs_filename}")
        self.outputs = torch.load(outputs_filename)

    @staticmethod
    def load(
        dirname: str, explorer: "Explorer", id: int, load_outputs: bool = True
    ) -> "Module":
        with open(Module.root_filename(dirname)) as f:
            data = json.load(f)
            name = data["name"]
            module = Module(explorer, id, name, None)
            if load_outputs:
                module.load_outputs(dirname)
            return module

    def to_dict(self, pc_samples: bool = True, ic_samples: bool = True):
        return {
            "id": self.id,
            "name": self.name,
            "pcs": [pc.to_dict(pc_samples) for pc in self.pcs],
            "ics": [ic.to_dict(ic_samples) for ic in self.ics],
        }

    def from_dict(self, data):
        self.name = data["name"]
        self.pcs = [
            Component.from_dict(pc_data, self, i)
            for (i, pc_data) in enumerate(data["pcs"])
        ]
        self.ics = [
            Component.from_dict(ic_data, self, i)
            for (i, ic_data) in enumerate(data["ics"])
        ]


class ModuleCorrelation:
    def __init__(self, a: Module, b: Module, value: float):
        self.a = a
        self.b = b
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.a.full_id.to_dict(),
            "b": self.b.full_id.to_dict(),
            "value": self.value,
        }
