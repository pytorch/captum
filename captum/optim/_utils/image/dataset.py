from typing import Dict, List, Optional

import torch

from captum.optim._utils.models import collect_activations
from captum.optim._utils.typing import ModuleOutputMapping


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
    a Karhunen-Loève transform (KLT) matrix, for a dataset.
    The color correlation matrix can then used in color decorrelation
    transforms for models trained on the dataset.
    """

    cov_mtx = dataset_cov_matrix(loader)
    return cov_matrix_to_klt(cov_mtx, normalize)


def capture_activation_samples(
    loader: torch.utils.data.DataLoader,
    model,
    targets: List[torch.nn.Module],
    target_names: List[str],
    num_samples: Optional[int] = None,
    input_device: torch.device = torch.device("cpu"),
) -> ModuleOutputMapping:
    """
    Create a dict of randomly sampled activations for an image dataset.

    Args:
        loader (torch.utils.data.DataLoader): A torch.utils.data.DataLoader
            instance for an image dataset.
        model (nn.Module): A PyTorch model instance.
        targets (list of nn.Module): A list of layers to sample activations
            from.
        target_names (list of str): A list of names to use for the layers
            to targets in the output dict.
        num_samples (int): How many samples to collect. Default is to collect
            all samples.
        input_device (torch.device): The device to use for model inputs.
    Returns:
        activation_dict (dict of tensor): A dictionary containing the sampled
            dataset activations, with the target_names as the keys.
    """

    def random_sample(activations: torch.Tensor) -> torch.Tensor:
        """
        Randomly sample H & W dimensions of activations with 4 dimensions.
        """

        rnd_samples = []
        for b in range(activations.size(0)):
            if activations.dim() == 4:
                h, w = activations.shape[2:]
                y = torch.randint(low=1, high=h, size=[1])
                x = torch.randint(low=1, high=w, size=[1])
                activ = activations[b, :, y, x]
            elif activations.dim() == 2:
                activ = activations[b].unsqueeze(1)
            rnd_samples.append(activ)
        return torch.cat(rnd_samples, 1).permute(1, 0)

    assert len(target_names) == len(targets)
    activation_dict: Dict = {k: [] for k in dict.fromkeys(target_names).keys()}

    sample_count = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(input_device)
            target_activ_dict = collect_activations(model, targets, inputs)
            for t in target_activ_dict.keys():
                target_activ_dict[t] = [random_sample(target_activ_dict[t])]
            activation_dict = {
                k: activation_dict[k] + target_activ_dict[k] for k in activation_dict
            }
            sample_count += inputs.size(0)
            if num_samples is not None:
                if sample_count > num_samples:
                    return {k: torch.cat(activation_dict[k]) for k in activation_dict}
    return {k: torch.cat(activation_dict[k]) for k in activation_dict}
