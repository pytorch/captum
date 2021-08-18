#!/usr/bin/env python3

import glob
import itertools
import os
from datetime import datetime
from typing import List, Tuple, Union

import torch
from captum.concept._core.concept import Concept
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class AV:
    r"""
    This class provides functionality to store and load activation vectors
    generated for pre-defined neural network layers and concepts.
    It also provides functionality to check if activation vectors already
    exist (are already stored) on the disk and other auxiliary functions.

    This class also defines a torch `Dataset`, representing Activation Vectors,
    which enables lazy access to activation vectors associated with a specific
    concept and layer stored on the disk.

    """

    r"""
        The name of the subfolder on the disk(storage) where the activation vectors
        are stored.
    """
    AV_DIR_NAME: str = "av"

    def __init__(self) -> None:
        pass

    class AVDataset(Dataset):
        r"""
        This dataset enables access to activation vectors for a given set of
        `concepts` in a given `layer` stored under a pre-defined path
        on the disk.
        The iterator of this dataset returns a batch of data tensors and
        corresponding labels. The batch size used here is the same used
        for `Concept's` data iterator.
        """

        def __init__(self, path: str, layer: str, concepts: List[Concept]):
            r"""
            Loads into memory the list of all activation file paths associated
            with the input `layer` and the list of concepts.

            Args:
                path (str): The path where the activation vectors
                        for the `layer` and `concepts` are stored.
                layer (str): The layer for which the activation vectors are
                        computed.
                concepts (list[Concept]): A list of concepts for which we
                        compute activation vectors in given input `layer`.

            """
            self.av_save_dir = AV._assemble_dir_path(path, layer)

            fls = [
                sorted(
                    glob.glob(AV._assemble_file_path(self.av_save_dir, concept, "*"))
                )
                for concept in concepts
            ]
            self.files = list(itertools.chain.from_iterable(fls))
            self.layer = layer

        def _extract_concept_id(self, fl_name: str) -> Union[None, int]:
            try:
                r_idx = fl_name.rfind("-")
                return int(fl_name[fl_name[:r_idx].rfind("-") + 1 : r_idx])
            except Exception:
                return None

        def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
            fl = self.files[idx]
            concept_id = self._extract_concept_id(fl)
            assert concept_id is not None, (
                "Couldn't extract valid concept_id" " from given file name: %s" % fl
            )
            inputs = torch.load(fl)
            labels = torch.tensor([concept_id] * inputs.size(0), device=inputs.device)
            return inputs, labels

        def __len__(self):
            return len(self.files)

    @staticmethod
    def _assemble_file_path(path: str, concept: Concept, suffix: str) -> str:
        return "-".join([path + concept.name, str(concept.id), suffix])

    @staticmethod
    def _assemble_dir_path(path: str, layer: str) -> str:
        return "/".join([path, AV.AV_DIR_NAME, layer, ""])

    @staticmethod
    def exists(path: str, concept: Concept, layer: str) -> bool:
        r"""
        Verifies whether the `concept` exists for given `layer`
        under the storage (disk) path.

        Args:
            path (str): The path where the activation vectors
                    for the `layer` and `concept` are stored.
            concept (Concept): A concept for which we
                    compute activation vectors in given input `layer`.
            layer (str): The layer for which the activation vectors are
                    computed.

        Returns:
            exists (bool): Indicating whether the `concept`'s activation
                    vectors for the `layer` were stored on the disk under
                    the `path`.

        """
        av_dir = AV._assemble_dir_path(path, layer)
        return (
            os.path.exists(av_dir)
            and len(glob.glob(AV._assemble_file_path(av_dir, concept, "*"))) > 0
        )

    @staticmethod
    def save(path: str, concept: Concept, layer: str, act_tensor: Tensor) -> None:
        r"""
        Saves the activation vectors `act_tensor` for the `concept` and
        `layer` under the data `path`.

        Args:
            path (str): The path where the activation vectors
                    for the `layer` and `concept` are stored.
            concept (Concept): A concept for which we
                    compute activation vectors in given input `layer`.
            layer (str): The layer for which the activation vectors are
                    computed.
            act_tensor (Tensor): A batch of activation vectors associated
                    with the `concept` and the `layer`.
        """
        av_dir = AV._assemble_dir_path(path, layer)

        if not os.path.exists(av_dir):
            os.makedirs(av_dir)

        av_save_fl_path = AV._assemble_file_path(
            av_dir, concept, str(int(datetime.now().microsecond))
        )
        torch.save(act_tensor, av_save_fl_path)

    @staticmethod
    # TODO fix num_workers: currently disabled since cav generation is
    # already parallelized
    def load(
        path: str, layer: str, concepts: List[Concept], num_workers: int = 0
    ) -> Union[None, DataLoader]:
        r"""
        Loads lazily the activation vectors for given `concepts` and
        `layer` saved under the `path`.

        Args:
            path (str): The path where the activation vectors
                    for the `layer` and `concept` are stored.
            concept (Concept): A concept for which we compute activation
                    vectors for given input `layer`.
            layer (str): The layer for which the activation vectors are
                    loaded.
            num_workers (int): The number of workers that are used for
                    distributing the load of reading activation vectors from
                    the data `path`.

        Returns:
            dataloader (DataLoader): A torch dataloader that allows to iterate
                    over the activation vectors for given layer and concepts.
        """

        def batch_collate(batch):
            inputs, labels = zip(*batch)
            return torch.cat(inputs), torch.cat(labels)

        assert num_workers == 0, (
            "Currently, the parallelization with multiple "
            "number of workers doesn't work because this functionality is being called "
            "from a worker / non-daemonic process."
        )

        av_save_dir = AV._assemble_dir_path(path, layer)

        if os.path.exists(av_save_dir):
            avdataset = AV.AVDataset(path, layer, concepts)
            loader = DataLoader(
                avdataset,
                collate_fn=batch_collate,
                num_workers=num_workers,
            )
            return loader
        return None
