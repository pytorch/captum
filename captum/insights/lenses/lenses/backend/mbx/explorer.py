#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import shutil
import json
import itertools
from tqdm import tqdm
from .sample import Sample
from typing import List, Optional, Any, Callable, Dict, TYPE_CHECKING


class ExplorerFullId:
    def __init__(self, id: int):
        self.id: int = id

    def to_dict(self) -> Dict[str, int]:
        return {"id": self.id}

    @staticmethod
    def from_dict(data: Dict[str, int]) -> "ExplorerFullId":
        return ExplorerFullId(data["id"])


from .module import Module
from .component import Component, Correlation as ComponentCorrelation, ComponentFullId

if TYPE_CHECKING:
    from .workspace import Workspace


class Explorer:
    def __init__(
        self,
        workspace: Optional["Workspace"] = None,
        id: int = 0,
        name: Optional[str] = None,
        get_sample: Optional[Callable[[int], Sample]] = None,
    ):
        self.workspace = workspace
        self.id = id
        if name is None:
            self.name = str(id)
        else:
            self.name = name
        self.modules: List[Module] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.pc_correlations: List[ComponentCorrelation] = []
        self.ic_correlations: List[ComponentCorrelation] = []

        if get_sample is not None:
            self.get_sample = get_sample
        else:
            default_sample = Sample()
            self.get_sample = lambda sample_id: default_sample

    @property
    def full_id(self) -> ExplorerFullId:
        return ExplorerFullId(self.id)

    def get_component_by_full_id(self, full_id: ComponentFullId) -> Component:
        module_id = full_id.module.id
        component_id = full_id.id
        return self.modules[module_id].pcs[component_id]

    def add_module(self, name: str, m: Optional[nn.Module]):
        id = len(self.modules)
        module = Module(self, id, name, m)
        self.modules.append(module)
        return module

    def register_hooks(self):
        for module in self.modules:
            m = module.m
            hook = m.register_forward_hook(module.forward_hook)
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_pcs(self, num_components: int = 5):
        for module in self.modules:
            module.compute_pcs(num_components)

    def compute_ics(self, num_components: int = 5):
        for module in self.modules:
            module.compute_ics(num_components)

    def compute_pc_correlations(self):
        self.pc_correlations = []
        for m1, m2 in itertools.combinations(self.modules, 2):
            for c1, c2 in itertools.product(m1.pcs, m2.pcs):
                self.pc_correlations.append(c1.compute_correlation(c2))

    def compute_ic_correlations(self):
        self.ic_correlations = []
        for m1, m2 in itertools.combinations(self.modules, 2):
            for c1, c2 in itertools.product(m1.ics, m2.ics):
                self.ic_correlations.append(c1.compute_correlation(c2))

    def to_dict(
        self, pc_samples: bool = True, ic_samples: bool = True
    ) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "modules": [
                module.to_dict(pc_samples, ic_samples) for module in self.modules
            ],
            "pc_correlations": [
                correlation.to_dict() for correlation in self.pc_correlations
            ],
            "ic_correlations": [
                correlation.to_dict() for correlation in self.ic_correlations
            ],
        }

    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        workspace: Optional["Workspace"] = None,
        id: int = 0,
        get_sample: Optional[Callable[[int], Sample]] = None,
    ) -> "Explorer":
        if "name" in data:
            name = data["name"]
        else:
            name = None

        explorer = Explorer(workspace, id, name, get_sample=get_sample)

        for module_data in data["modules"]:
            name = module_data["name"]
            module = explorer.add_module(name, None)
            module.from_dict(module_data)

        explorer.pc_correlations = [
            ComponentCorrelation.from_dict(correlation_data, explorer)
            for correlation_data in data["pc_correlations"]
        ]

        explorer.ic_correlations = [
            ComponentCorrelation.from_dict(correlation_data, explorer)
            for correlation_data in data["ic_correlations"]
        ]

        return explorer

    @staticmethod
    def root_filename(dirname: str) -> str:
        return os.path.join(dirname, "explorer")

    @staticmethod
    def modules_dirname(dirname: str) -> str:
        return os.path.join(dirname, "modules")

    def save(self, dirname: str, overwrite: bool = False, save_outputs: bool = True):
        if os.path.exists(dirname):
            if not overwrite:
                raise RuntimeError(
                    f"{dirname} exists, to overwrite run Explorer.save({dirname}, overwrite=True)"
                )

            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
            else:
                os.remove(dirname)

        os.makedirs(dirname)
        with open(Explorer.root_filename(dirname), "w") as f:
            json.dump(self.to_dict(), f)

        modules_dirname = self.modules_dirname(dirname)
        os.makedirs(modules_dirname)
        for module in self.modules:
            module_dirname = os.path.join(modules_dirname, str(module.id))
            module.save(module_dirname, save_outputs=save_outputs)

    @staticmethod
    def load(
        dirname: str,
        workspace: Optional["Workspace"] = None,
        id: int = 0,
        name: Optional[str] = None,
        get_sample: Optional[Callable[[int], Sample]] = None,
        load_module_outputs: bool = True,
    ) -> "Explorer":
        with open(Explorer.root_filename(dirname)) as f:
            data = json.load(f)
            explorer = Explorer.from_dict(data, workspace, id, get_sample)

        if load_module_outputs:
            modules_dirname = explorer.modules_dirname(dirname)
            for module in tqdm(explorer.modules):
                module_dirname = os.path.join(modules_dirname, str(module.id))
                module.load_outputs(module_dirname)

        return explorer

    def run(
        self,
        forward: Callable[..., Any],
        num_components: int = 5,
        only_outputs: bool = False,
    ):
        try:
            self.register_hooks()
            forward()
            if not only_outputs:
                self.compute_pcs(num_components)
                self.compute_ics(num_components)
                self.compute_pc_correlations()
                self.compute_ic_correlations()
        finally:
            self.remove_hooks()
