from typing import cast

import torch

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " image dataset functions with progress bar"
    )


def image_cov(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor's RGB covariance matrix
    """

    assert x.dim() > 1
    x = x.reshape(-1, x.size(1)).T
    x = x - torch.mean(x, dim=-1).unsqueeze(-1)
    return 1 / (x.shape[-1] - 1) * x @ x.transpose(-1, -2)


def dataset_cov_matrix(
    loader: torch.utils.data.DataLoader,
    show_progress: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Calculate the covariance matrix for an image dataset.
    """

    if show_progress:
        pbar = tqdm(total=len(loader.dataset), unit=" images")  # type: ignore

    cov_mtx = cast(torch.Tensor, 0.0)
    for images, _ in loader:
        assert images.dim() > 1
        images = images.to(device)
        cov_mtx = cov_mtx + image_cov(images)
        if show_progress:
            pbar.update(images.size(0))

    if show_progress:
        pbar.close()

    cov_mtx = cov_mtx / cast(int, len(loader.dataset))
    return cov_mtx


def cov_matrix_to_klt(
    cov_mtx: torch.Tensor, normalize: bool = False, epsilon: float = 1e-10
) -> torch.Tensor:
    """
    Convert a cov matrix to a klt matrix.
    """

    U, S, V = torch.svd(cov_mtx)
    svd_sqrt = U @ torch.diag(torch.sqrt(S + epsilon))
    if normalize:
        svd_sqrt / torch.max(torch.norm(svd_sqrt, dim=0))
    return svd_sqrt


def dataset_klt_matrix(
    loader: torch.utils.data.DataLoader,
    normalize: bool = False,
    show_progress: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Calculate the color correlation matrix, also known as
    a Karhunen-Loève transform (KLT) matrix, for a dataset.
    The color correlation matrix can then used in color decorrelation
    transforms for models trained on the dataset.
    """

    cov_mtx = dataset_cov_matrix(loader, show_progress=show_progress, device=device)
    return cov_matrix_to_klt(cov_mtx, normalize)
