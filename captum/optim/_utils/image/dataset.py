import os
from typing import Dict, List, Optional

import torch

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " image dataset functions with progress bar"
    )

from captum.optim._utils.models import collect_activations


def image_cov(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor's RGB covariance matrix
    """

    tensor = tensor.reshape(-1, 3)
    tensor = tensor - tensor.mean(0, keepdim=True)
    return 1 / (tensor.size(0) - 1) * tensor.T @ tensor


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
        assert images.dim() == 4
        images = images.to(device)
        for b in range(images.size(0)):
            cov_mtx = cov_mtx + image_cov(images[b].permute(1, 2, 0))

            if show_progress:
                pbar.update(1)

    if show_progress:
        pbar.close()

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


def capture_activation_samples(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    targets: List[torch.nn.Module],
    target_names: Optional[List[str]] = None,
    sample_dir: str = "",
    num_images: Optional[int] = None,
    samples_per_image: int = 1,
    input_device: torch.device = torch.device("cpu"),
    show_progress: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Capture randomly sampled activations for an image dataset from one or multiple
    target layers.
    Args:
        loader (torch.utils.data.DataLoader): A torch.utils.data.DataLoader
            instance for an image dataset.
        model (nn.Module): A PyTorch model instance.
        targets (list of nn.Module): A list of layers to collect activation samples
            from.
        target_names (list of str, optional): A list of names to use when saving sample
            tensors as files.
        sample_dir (str): Path to where activation samples should be saved.
        num_images (int, optional): How many images to collect samples from.
            Default is to collect samples for every image in the dataset.
        samples_per_image (int): How many samples to collect per image. Default
            is 1 sample per image.
        input_device (torch.device, optional): The device to use for model
            inputs.
        show_progress (bool, optional): Whether or not to show progress.
    """

    def random_sample(activations: torch.Tensor) -> torch.Tensor:
        """
        Randomly sample H & W dimensions of activations with 4 dimensions.
        """
        assert activations.dim() == 4 or activations.dim() == 2

        rnd_samples = []
        for i in range(samples_per_image):
            for b in range(activations.size(0)):
                if activations.dim() == 4:
                    h, w = activations.shape[2:]
                    y = torch.randint(low=1, high=h - 1, size=[1])
                    x = torch.randint(low=1, high=w - 1, size=[1])
                    activ = activations[b, :, y, x]
                elif activations.dim() == 2:
                    activ = activations[b].unsqueeze(1)
                rnd_samples.append(activ)
        return rnd_samples

    if target_names is None:
        target_names = ["target" + str(i) + "_" for i in range(len(targets))]
    assert len(target_names) == len(targets)
    assert os.path.isdir(sample_dir)

    if show_progress:
        total = (
            len(loader.dataset) if num_images is None else num_images  # type: ignore
        )
        pbar = tqdm(total=total, unit=" images")

    image_count, batch_count = 0, 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(input_device)
            image_count += inputs.size(0)
            batch_count += 1

            target_activ_dict = collect_activations(model, targets, inputs)

            [
                torch.save(
                    random_sample(target_activ_dict[t]),
                    os.path.join(sample_dir, +n + "_" + str(batch_count) + ".pt"),
                )
                for t, n in zip(target_activ_dict, target_names)
            ]
            del target_activ_dict

            if show_progress:
                pbar.update(inputs.size(0))

            if num_images is not None:
                if image_count > num_images:
                    break

    if show_progress:
        pbar.close()


def consolidate_samples(
    sample_dir: str = "samples", sample_basename: str = "", show_progress: bool = False
) -> torch.Tensor:
    """
    Combine samples collected from capture_activation_samples into a single
    tensor with a shape of [n_samples, n_channels].

    Args:
        sample_dir (str): The directory where activation samples where saved.
        sample_basename (str, optional): If samples from different layers are
            present in sample_dir, then you can use samples from only a
            specific layer by specifying the basename that samples of the same
            layer share.
        show_progress (bool, optional): Whether or not to show progress.
    Returns:
        sample_tensor (torch.Tensor): A tensor containing all the specified
            sample tensors with a shape of [n_samples, n_channels].
    """

    samples = []
    tensor_samples = [
        os.path.join(sample_dir, name)
        for name in os.listdir(sample_dir)
        if sample_basename.lower() in name.lower()
        and os.path.isfile(os.path.join(sample_dir, name))
    ]

    if show_progress:
        pbar = tqdm(total=len(tensor_samples), unit=" sample batches collected")
    for file in tensor_samples:
        sample_batch = torch.load(file)
        for s in sample_batch:
            samples += [s.cpu()]
        if show_progress:
            pbar.update(1)
    if show_progress:
        pbar.close()
    return torch.cat(samples, 1).permute(1, 0)
