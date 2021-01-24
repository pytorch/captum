#!/usr/bin/env python3
import os
import unittest

import torch

import captum.optim._utils.image.dataset as dataset_utils
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.optim.helpers import image_dataset as dataset_helpers


class TestImageCov(BaseTest):
    def test_image_cov(self) -> None:
        test_tensor = torch.cat(
            [
                torch.ones(1, 4, 4) * 0.1,
                torch.ones(1, 4, 4) * 0.2,
                torch.ones(1, 4, 4) * 0.3,
            ],
            0,
        )

        output_tensor = dataset_utils.image_cov(test_tensor)
        expected_output = dataset_helpers.image_cov_np(test_tensor.numpy())
        assertArraysAlmostEqual(output_tensor.numpy(), expected_output, 0.01)


class TestDatasetCovMatrix(BaseTest):
    def test_dataset_cov_matrix(self) -> None:
        num_tensors = 100

        def create_tensor() -> torch.Tensor:
            return torch.cat(
                [
                    torch.ones(1, 224, 224) * 0.1,
                    torch.ones(1, 224, 224) * 0.2,
                    torch.ones(1, 224, 224) * 0.3,
                ],
                0,
            )

        dataset_tensors = [create_tensor() for x in range(num_tensors)]
        test_dataset = dataset_helpers.ImageTestDataset(dataset_tensors)
        dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, num_workers=0, shuffle=False
        )
        output_mtx = dataset_utils.dataset_cov_matrix(dataset_loader)
        expected_mtx = torch.tensor(
            [
                [4.9961e-14, 9.9922e-14, -6.6615e-14],
                [9.9922e-14, 1.9984e-13, -1.3323e-13],
                [-6.6615e-14, -1.3323e-13, 8.8820e-14],
            ]
        )
        assertTensorAlmostEqual(self, output_mtx, expected_mtx)


class TestCovMatrixToKLT(BaseTest):
    def test_cov_matrix_to_klt(self) -> None:
        test_input = torch.tensor(
            [
                [0.0477, 0.0415, 0.0280],
                [0.0415, 0.0425, 0.0333],
                [0.0280, 0.0333, 0.0419],
            ]
        )
        output_mtx = dataset_utils.cov_matrix_to_klt(test_input)
        expected_mtx = dataset_helpers.cov_matrix_to_klt_np(test_input.numpy())
        assertArraysAlmostEqual(output_mtx.numpy(), expected_mtx, 0.0005)


class TestDatasetKLTMatrix(BaseTest):
    def test_dataset_klt_matrix(self) -> None:
        num_tensors = 100

        def create_tensor() -> torch.Tensor:
            return torch.cat(
                [
                    torch.ones(1, 224, 224) * 0.1,
                    torch.ones(1, 224, 224) * 0.2,
                    torch.ones(1, 224, 224) * 0.3,
                ],
                0,
            )

        dataset_tensors = [create_tensor() for x in range(num_tensors)]
        test_dataset = dataset_helpers.ImageTestDataset(dataset_tensors)
        dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, num_workers=0, shuffle=False
        )

        klt_transform = dataset_utils.dataset_klt_matrix(dataset_loader)

        expected_mtx = torch.tensor(
            [
                [-3.8412e-06, 9.2125e-06, 6.1284e-07],
                [-7.6823e-06, -3.5571e-06, 5.3226e-06],
                [5.1216e-06, 1.5737e-06, 8.4436e-06],
            ]
        )

        assertTensorAlmostEqual(self, klt_transform, expected_mtx)


class TestCaptureActivationSamples(BaseTest):
    def test_capture_activation_samples(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping capture_activation_samples test due to"
                + "insufficient Torch version."
            )

        num_tensors = 20
        dataset_tensors = [torch.ones(3, 224, 224) for x in range(num_tensors)]
        test_dataset = dataset_helpers.ImageTestDataset(dataset_tensors)
        dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=5, num_workers=0, shuffle=False
        )
        model = googlenet(pretrained=True)
        targets = [model.mixed4c]
        target_names = ["mixed4c"]
        sample_dir = "test_samples"
        os.mkdir(sample_dir)

        dataset_utils.capture_activation_samples(
            dataset_loader, model, targets, target_names
        )

        tensor_samples_files = [
            os.path.join(sample_dir, name)
            for name in os.listdir(sample_dir)
            if os.path.isfile(os.path.join(sample_dir, name))
        ]
        tensor_samples = []
        [tensor_samples + torch.load(file) for file in tensor_samples_files]
        sample_tensor = torch.cat(tensor_samples, 1).permute(1, 0)
        self.assertEqual(list(sample_tensor.shape), [num_tensors, 512])


class TestConsolidateSamples(BaseTest):
    def test_consolidate_samples(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping consolidate_samples test due to"
                + "insufficient Torch version."
            )

        sample_dir = "test_samples_consolidation"
        os.mkdir(sample_dir)
        num_channels = 512
        num_files = 10
        batch_size = 4
        for i, f in enumerate(num_files):
            tensor_batch = [torch.ones(num_channels, 1) for x in range(batch_size)]
            torch.save(
                tensor_batch, os.path.join(sample_dir, "tensor_batch_" + str(i) + ".pt")
            )

        sample_tensor = dataset_utils.consolidate_samples(sample_dir)
        self.assertEqual(
            list(sample_tensor.shape), [num_files * batch_size, num_channels]
        )


if __name__ == "__main__":
    unittest.main()
