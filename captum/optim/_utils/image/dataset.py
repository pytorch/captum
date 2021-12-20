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
    Calculate a tensor's RGB covariance matrix.

    Args:

        tensor (tensor):  An NCHW image tensor.

    Returns:
        *tensor*:  An RGB covariance matrix for the specified tensor.
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

    Args:

        loader (torch.utils.data.DataLoader):  The reference to a PyTorch
            dataloader instance.
        show_progress (bool, optional): Whether or not to display a tqdm progress bar.
            Default: False
        device (torch.device, optional): The PyTorch device to use for for calculating
            the cov matrix.
            Default: torch.device("cpu")

    Returns:
        *tensor*:  A covariance matrix for the specified dataset.
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

    Args:

        cov_mtx (tensor):  A 3 by 3 covariance matrix generated from a dataset.
        normalize (bool):  Whether or not to normalize the resulting KLT matrix.
            Default: False
        epsilon (float):

    Returns:
        *tensor*:  A KLT matrix for the specified covariance matrix.
    """

    # Handle older versions of PyTorch
    torch_norm = torch.linalg.norm if torch.__version__ >= "1.9.0" else torch.norm

    U, S, V = torch.svd(cov_mtx)
    svd_sqrt = U @ torch.diag(torch.sqrt(S + epsilon))
    if normalize:
        svd_sqrt / torch.max(torch_norm(svd_sqrt, dim=0))
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

    Args:

        loader (torch.utils.data.DataLoader):  The reference to a PyTorch
            dataloader instance.
        normalize (bool):  Whether or not to normalize the resulting KLT matrix.
            Default: False
        show_progress (bool, optional): Whether or not to display a tqdm progress bar.
            Default: False
        device (torch.device, optional): The PyTorch device to use for for calculating
            the cov matrix.
            Default: torch.device("cpu")

    Returns:
        *tensor*:  A KLT matrix for the specified dataset.
    """

    cov_mtx = dataset_cov_matrix(loader, show_progress=show_progress, device=device)
    return cov_matrix_to_klt(cov_mtx, normalize)
