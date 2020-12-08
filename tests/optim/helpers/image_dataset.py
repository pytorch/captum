from typing import List, Tuple

import numpy as np
import torch


class ImageTestDataset(torch.utils.data.Dataset):
    def __init__(self, tensors: List[torch.Tensor]) -> None:
        assert all(t.size(0) == 1 for t in tensors if t.dim() == 4)

        def t_squeeze(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0) if x.dim() == 4 else x

        tensors = [t_squeeze(t) for t in tensors]
        self.tensors = tensors

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.tensors[idx], 0

    def __len__(self) -> int:
        return len(self.tensors)


def image_cov_np(array: np.ndarray) -> np.ndarray:
    """
    Calculate an array's RGB covariance matrix
    """

    array = array.reshape(-1, 3)
    array = array - array.mean(0, keepdims=True)
    return 1 / (array.shape[0] - 1) * array.T @ array


def cov_matrix_to_klt_np(
    cov_mtx: np.ndarray, normalize: bool = False, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Convert a cov matrix to a klt matrix.
    """

    U, S, V = np.linalg.svd(cov_mtx)
    svd_sqrt = U @ np.diag(np.sqrt(S + epsilon))
    if normalize:
        svd_sqrt / np.linalg.norm(svd_sqrt, axis=0).max()
    return svd_sqrt
