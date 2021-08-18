import glob
import tempfile
from typing import cast

import torch
from captum.concept._core.av import AV
from captum.concept._core.concept import Concept
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from torch.utils.data import DataLoader


class Test(BaseTest):
    def test_av_save_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            concept0 = self._create_concept(0, "test_concept_0")
            concept0_batch0 = torch.randn(64, 16)

            AV.save(tmpdir, concept0, "layer1", concept0_batch0)
            self.assertTrue(AV.exists(tmpdir, concept0, "layer1"))
            self.assertFalse(AV.exists(tmpdir, concept0, "layer2"))

            # experimenting with a new concept
            concept1 = self._create_concept(1, "test_concept_2")
            concept1_batch0 = torch.randn(64, 16)

            self.assertFalse(AV.exists(tmpdir, concept1, "layer1"))
            AV.save(tmpdir, concept1, "layer1", concept1_batch0)
            self.assertTrue(AV.exists(tmpdir, concept1, "layer1"))

    def test_av_save_multi_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            concept0 = self._create_concept(0, "test_concept_0")
            concept0_batch0 = torch.randn(64, 16)
            concept0_batch1 = torch.randn(64, 16)
            concept0_batch2 = torch.randn(32, 16)

            # test first batch
            AV.save(tmpdir, concept0, "layer1", concept0_batch0)
            concept0_batch0_path = AV._assemble_file_path(
                AV._assemble_dir_path(tmpdir, "layer1"), concept0, "*"
            )
            self.assertEqual(len(glob.glob(concept0_batch0_path)), 1)

            # test second batch
            AV.save(tmpdir, concept0, "layer1", concept0_batch1)
            concept0_batch1_path = AV._assemble_file_path(
                AV._assemble_dir_path(tmpdir, "layer1"), concept0, "*"
            )
            self.assertEqual(len(glob.glob(concept0_batch1_path)), 2)

            # test third batch
            AV.save(tmpdir, concept0, "layer1", concept0_batch2)
            concept0_batch2_path = AV._assemble_file_path(
                AV._assemble_dir_path(tmpdir, "layer1"), concept0, "*"
            )
            self.assertEqual(len(glob.glob(concept0_batch2_path)), 3)

    def test_av_load_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            concept0 = self._create_concept(0, "test_concept_0")
            concept0_batch0 = torch.randn(64, 16)
            concept1 = self._create_concept(1, "test_concept_1")
            concept1_batch0 = torch.randn(36, 16)

            AV.save(tmpdir, concept0, "layer1", concept0_batch0)
            dataloader = AV.load(tmpdir, "layer1", [concept0])
            self.assertIsNotNone(dataloader)
            for input, label in cast(DataLoader, dataloader):
                assertTensorAlmostEqual(self, input, concept0_batch0)
                assertTensorAlmostEqual(
                    self, label.float(), torch.zeros(input.size(0)).float()
                )

            # add concept1 to the list of concepts
            AV.save(tmpdir, concept1, "layer1", concept1_batch0)
            dataloader = cast(
                DataLoader, AV.load(tmpdir, "layer1", [concept0, concept1])
            )
            self.assertIsNotNone(dataloader)
            concepts_batch = [concept0_batch0, concept1_batch0]
            for i, (input, label) in enumerate(dataloader):
                assertTensorAlmostEqual(self, input, concepts_batch[i])
                assertTensorAlmostEqual(
                    self, label.float(), (torch.ones(input.size(0)) * i).float()
                )

    def test_av_load_multi_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            concept0 = self._create_concept(0, "test_concept_0")
            concept0_batch0 = torch.randn(64, 16)
            concept0_batch1 = torch.randn(64, 16)
            concept0_batch2 = torch.randn(32, 16)

            # save all batches for concept0
            AV.save(tmpdir, concept0, "layer1", concept0_batch0)
            AV.save(tmpdir, concept0, "layer1", concept0_batch1)
            AV.save(tmpdir, concept0, "layer1", concept0_batch2)

            concept_batches = [concept0_batch0, concept0_batch1, concept0_batch2]
            dataloader = cast(DataLoader, AV.load(tmpdir, "layer1", [concept0]))
            self.assertIsNotNone(dataloader)
            for i, (input, label) in enumerate(dataloader):
                assertTensorAlmostEqual(self, input, concept_batches[i])
                assertTensorAlmostEqual(
                    self, label.float(), torch.zeros(input.size(0)).float()
                )

    def test_av_load_non_saved_concept(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            concept0 = self._create_concept(0, "test_concept_0")
            dataloader = AV.load(tmpdir, "layer1", [concept0])
            self.assertIsNone(dataloader)

    def _create_concept(self, id: int, name: str) -> Concept:
        return Concept(id, name, None)
