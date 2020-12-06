#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.datasets as dataset_utils
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.optim.helpers import datasets as dataset_helpers


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


if __name__ == "__main__":
    unittest.main()
