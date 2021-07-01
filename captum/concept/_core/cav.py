#!/usr/bin/env python3

import os
from typing import Any, Dict, List

import torch

from captum.concept._core.concept import Concept
from captum.concept._utils.common import concepts_to_str


class CAV:
    r"""
    Concept Activation Vector (CAV) is a vector orthogonal to the decision
    boundary of a classifier which distinguishes between activation
    vectors produced by different concepts.
    More details can be found in the paper:
        https://arxiv.org/pdf/1711.11279.pdf
    """

    def __init__(
        self,
        concepts: List[Concept],
        layer: str,
        stats: Dict[str, Any] = None,
        save_path: str = "./cav/",
    ) -> None:
        r"""
        This class encapsulates the instances of CAVs objects, saves them in
        and loads them from the disk (storage).

        Args:
            concepts (list[Concept]): a List of Concept objects. Only their
                        names will be saved and loaded.
            layer (str): The layer where concept activation vectors are
                        computed using a predefined classifier.
            stats (dict): a dictionary that retains information about the CAV
                        classifier such as CAV weights and accuracies.
                        Ex.: stats = {"weights": weights, "classes": classes,
                                      "accs": accs}, where "weights" are learned
                        model parameters, "classes" are a list of classes used
                        by the model to generate the "weights" and "accs"
                        the classifier training or validation accuracy.
            save_path (str): The path where the CAV objects are stored.

        """

        self.concepts = concepts
        self.layer = layer
        self.stats = stats
        self.save_path = save_path

    @staticmethod
    def assemble_save_path(path: str, concepts: List[Concept], layer: str):
        r"""
        A utility method for assembling filename and its path, from
        a concept list and a layer name.

        Args:
            path (str): A path to be concatenated with the concepts key and
                    layer name.
            concepts (list(Concept)): A list of concepts that are concatenated
                    together and used as a concept key using their ids. These
                    concept ids are retrieved from TCAV s`Concept` objects.
            layer (str): The name of the layer for which the activations are
                    computed.

        Returns:
            cav_path(str): A string containing the path where the computed CAVs
                    will be stored.
                    For example, given:
                        concept_ids = [0, 1, 2]
                        concept_names = ["striped", "random_0", "random_1"]
                        layer = "inception4c"
                        path = "/cavs",
                    the resulting save path will be:
                        "/cavs/0-1-2-inception4c.pkl"

        """

        file_name = concepts_to_str(concepts) + "-" + layer + ".pkl"

        return os.path.join(path, file_name)

    def save(self):
        r"""
        Saves a dictionary of the CAV computed values into a pickle file in the
        location returned by the "assemble_save_path" static methods. The
        dictionary contains the concept names list, the layer name for which
        the activations are computed for, the stats dictionary which contains
        information about the classifier train/eval statistics such as the
        weights and training accuracies. Ex.:

        save_dict = {
            "concept_ids": [0, 1, 2],
            "concept_names": ["striped", "random_0", "random_1"],
            "layer": "inception4c",
            "stats": {"weights": weights, "classes": classes, "accs": accs}
        }

        """

        save_dict = {
            "concept_ids": [c.id for c in self.concepts],
            "concept_names": [c.name for c in self.concepts],
            "layer": self.layer,
            "stats": self.stats,
        }

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        cavs_path = CAV.assemble_save_path(self.save_path, self.concepts, self.layer)

        torch.save(save_dict, cavs_path)

    @staticmethod
    def load(cavs_path: str, concepts: List[Concept], layer: str):
        r"""
        Loads CAV dictionary from a pickle file for given input
        `layer` and `concepts`.

        Args:
            cavs_path (str): The root path where the cavs are stored
                    in the storage (on the disk).
                    Ex.: "/cavs"
            concepts (list[Concept]):  A List of concepts for which
                    we would like to load the cavs.
            layer (str): The layer name. Ex.: "inception4c". In case of nested
                    layers we use dots to specify the depth / hierarchy.
                    Ex.: "layer.sublayer.subsublayer"

        Returns:
            cav(CAV): An instance of a CAV class, containing the respective CAV
                    score per concept and layer. An example of a path where the
                    cavs are loaded from is:
                    "/cavs/0-1-2-inception4c.pkl"
        """

        cavs_path = CAV.assemble_save_path(cavs_path, concepts, layer)

        if os.path.exists(cavs_path):
            save_dict = torch.load(cavs_path)

            concept_names = save_dict["concept_names"]
            concept_ids = save_dict["concept_ids"]
            concepts = [
                Concept(concept_id, concept_name, None)
                for concept_id, concept_name in zip(concept_ids, concept_names)
            ]
            cav = CAV(concepts, save_dict["layer"], save_dict["stats"])

            return cav

        return None
