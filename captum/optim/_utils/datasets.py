import torch


def image_cov(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor's RGB covariance matrix
    """

    tensor = tensor.reshape(-1, 3)
    tensor = tensor - tensor.mean(0, keepdim=True)
    return 1 / (tensor.size(0) - 1) * tensor.T @ tensor


def dataset_cov_matrix(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Calculate the covariance matrix for an image dataset.
    """

    cov_mtx = torch.zeros(3, 3)
    for images, _ in loader:
        assert images.dim() == 4
        for b in range(images.size(0)):
            cov_mtx = cov_mtx + image_cov(images[b].permute(1, 2, 0))
    cov_mtx = cov_mtx / len(loader.dataset)  # type: ignore
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
    loader: torch.utils.data.DataLoader, normalize: bool = False
) -> torch.Tensor:
    """
    Calculate the color correlation matrix, also known as
    a Karhunen-Loève transform (KLT), for a dataset.
    The color correlation matrix can then used in color decorrelation
    transforms for models trained on the dataset.
    """

    cov_mtx = dataset_cov_matrix(loader)
    return cov_matrix_to_klt(cov_mtx, normalize)
