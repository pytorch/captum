from .explorer import ExplorerFullId, Explorer
from .module import ModuleFullId, ModuleCorrelation, Module
from .component import Component, ComponentFullId, Correlation as ComponentCorrelation
from .sample import Sample
import itertools
import json
import os
import shutil
from tqdm import tqdm
import torch
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any

root_filename = "root.json"
explorers_dirname = "explorers"


class Workspace:
    def __init__(self, id: Optional[str] = None):
        if id is None:
            id = datetime.now().isoformat()
        self.id: str = id
        self.explorers: List[Explorer] = []
        self.module_pc_correlations: List[ComponentCorrelation] = []
        self.module_ic_correlations: List[ComponentCorrelation] = []
        self.module_correlations: List[ModuleCorrelation] = []

    def add_explorer(
        self,
        name: Optional[str] = None,
        get_sample: Optional[Callable[[int], Sample]] = None,
    ):
        id = len(self.explorers)
        explorer = Explorer(self, id, name, get_sample)
        self.explorers.append(explorer)
        return explorer

    def load_explorer(
        self,
        dirname: str,
        name: Optional[str] = None,
        get_sample: Optional[Callable[[int], Sample]] = None,
        load_module_outputs: bool = True,
    ):
        id = len(self.explorers)
        explorer = Explorer.load(
            dirname, self, id, name, get_sample, load_module_outputs
        )
        self.explorers.append(explorer)
        return explorer

    def get_explorer_by_full_id(self, full_id: ExplorerFullId) -> Explorer:
        return self.explorers[full_id.id]

    def get_module_by_full_id(self, full_id: ModuleFullId) -> Module:
        explorer = self.get_explorer_by_full_id(full_id.explorer)
        return explorer.modules[full_id.id]

    def get_pc_by_full_id(self, full_id: ComponentFullId) -> Component:
        component_id = full_id.id
        module = self.get_module_by_full_id(full_id.module)
        return module.pcs[component_id]

    def get_ic_by_full_id(self, full_id: ComponentFullId) -> Component:
        component_id = full_id.id
        module = self.get_module_by_full_id(full_id.module)
        return module.ics[component_id]

    def compute_correlations(self, cca: bool = False, cca_components: int = 5):
        self.compute_module_pc_correlations()
        self.compute_module_ic_correlations()
        if cca:
            self.compute_module_correlations(cca_components)

    def compute_for_module_pairs(
        self, f: Callable[[Module, Module], Any], verbose: bool = True
    ):
        # iterate over module pairs of different explorer instances
        explorer_pairs = itertools.combinations(self.explorers, 2)
        if verbose:
            explorer_pairs = tqdm(
                list(explorer_pairs), bar_format="{l_bar}{bar}{r_bar} [explorer pairs]"
            )
        for e1, e2 in explorer_pairs:
            module_pairs = itertools.product(e1.modules, e2.modules)
            if verbose:
                module_pairs = tqdm(
                    list(module_pairs), bar_format="{l_bar}{bar}{r_bar} [module pairs]"
                )
            for m1, m2 in module_pairs:
                f(m1, m2)

    def compute_module_pc_correlations(self, verbose: bool = True):
        self.module_pc_correlations = []
        if verbose:
            print("computing module PC correlations...")

        def compute_pc_correlations(m1, m2):
            for c1, c2 in itertools.product(m1.pcs, m2.pcs):
                self.module_pc_correlations.append(c1.compute_correlation(c2))

        self.compute_for_module_pairs(compute_pc_correlations, verbose)

    def compute_module_ic_correlations(self, verbose: bool = True):
        self.module_ic_correlations = []
        if verbose:
            print("computing module IC correlations...")

        def compute_ic_correlations(m1, m2):
            for c1, c2 in itertools.product(m1.ics, m2.ics):
                self.module_ic_correlations.append(c1.compute_correlation(c2))

        self.compute_for_module_pairs(compute_ic_correlations, verbose)

    def compute_module_correlations(
        self,
        num_components: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        self.module_correlations = []

        explorer_pairs = itertools.combinations(self.explorers, 2)
        if verbose:
            print("computing module correlations...")
            explorer_pairs = tqdm(
                list(explorer_pairs), bar_format="{l_bar}{bar}{r_bar} [explorer pairs]"
            )
        for e1, e2 in explorer_pairs:
            module_pairs = itertools.product(e1.modules, e2.modules)
            if verbose:
                module_pairs = tqdm(
                    list(module_pairs), bar_format="{l_bar}{bar}{r_bar} [module pairs]"
                )
            for m1, m2 in module_pairs:
                self.module_correlations.append(
                    m1.compute_correlation(m2, num_components, max_iter, tol)
                )

    def to_dict(self) -> Dict[str, Any]:
        explorers_data = []
        for explorer in self.explorers:
            explorers_data.append({"id": explorer.id})

        return {
            "id": self.id,
            "explorers": explorers_data,
            "module_correlations": [c.to_dict() for c in self.module_correlations],
            "module_pc_correlations": [
                c.to_dict() for c in self.module_pc_correlations
            ],
            "module_ic_correlations": [
                c.to_dict() for c in self.module_ic_correlations
            ],
        }

    def save(
        self, dirname: str, overwrite: bool = False, save_module_outputs: bool = True
    ):
        if os.path.exists(dirname):
            if not overwrite:
                raise RuntimeError(
                    f"{dirname} exists, to overwrite run Workspace.save({dirname}, overwrite=True)"
                )

            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
            else:
                os.remove(dirname)

        os.makedirs(dirname)
        with open(os.path.join(dirname, root_filename), "w") as f:
            json.dump(self.to_dict(), f)

        os.makedirs(os.path.join(dirname, explorers_dirname))

        for explorer in self.explorers:
            explorer_dirname = os.path.join(
                dirname, explorers_dirname, str(explorer.id)
            )
            explorer.save(explorer_dirname, save_outputs=save_module_outputs)

    @staticmethod
    def load(
        dirname: str,
        refresh_id: bool = True,
        get_sample: Optional[Callable[[int], Sample]] = None,
        load_module_outputs: bool = False,
    ) -> "Workspace":
        with open(os.path.join(dirname, root_filename)) as f:
            data = json.load(f)

            id: Optional[str] = None
            if not refresh_id and "id" in data:
                id = data["id"]

            workspace = Workspace(id)

            for explorer_data in data["explorers"]:
                explorer_id = explorer_data["id"]
                workspace.load_explorer(
                    os.path.join(dirname, explorers_dirname, str(explorer_id)),
                    get_sample=get_sample,
                    load_module_outputs=load_module_outputs,
                )

            for c_data in data["module_pc_correlations"]:
                pc_a = workspace.get_pc_by_full_id(
                    ComponentFullId.from_dict(c_data["a"])
                )
                pc_b = workspace.get_pc_by_full_id(
                    ComponentFullId.from_dict(c_data["b"])
                )
                value = c_data["value"]
                workspace.module_pc_correlations.append(
                    ComponentCorrelation(pc_a, pc_b, value)
                )

            for c_data in data["module_ic_correlations"]:
                ic_a = workspace.get_ic_by_full_id(
                    ComponentFullId.from_dict(c_data["a"])
                )
                ic_b = workspace.get_ic_by_full_id(
                    ComponentFullId.from_dict(c_data["b"])
                )
                value = c_data["value"]
                workspace.module_pc_correlations.append(
                    ComponentCorrelation(ic_a, ic_b, value)
                )

            for c_data in data["module_correlations"]:
                module_a = workspace.get_module_by_full_id(
                    ModuleFullId.from_dict(c_data["a"])
                )
                module_b = workspace.get_module_by_full_id(
                    ModuleFullId.from_dict(c_data["b"])
                )
                value = c_data["value"]
                workspace.module_correlations.append(
                    ModuleCorrelation(module_a, module_b, value)
                )

        return workspace

    @staticmethod
    def from_embeddings(
        embeddings: List[torch.Tensor],
        get_sample: Optional[Callable[[int], Sample]] = None,
    ):
        workspace = Workspace()
        explorer = workspace.add_explorer(get_sample=get_sample)
        embedding = explorer.add_module("embedding", None)
        embedding.outputs = embeddings
        embedding.compute_pcs()
        embedding.compute_ics()
        explorer.compute_pc_correlations()
        explorer.compute_ic_correlations()
        return workspace

    def show(self, dev_frontend_host: Optional[str] = None):
        r"""
            Invoke this method to start a web application which
            can be used to interactively explore the embeddings.
        """
        from lenses.backend.web_server import WebServer
        from lenses.backend.mbx.embedding_explorer_service import (
            EmbeddingExplorerService,
        )

        service = EmbeddingExplorerService(self)
        web_server = WebServer(service, dev_frontend_host=dev_frontend_host)
        web_server.start()
