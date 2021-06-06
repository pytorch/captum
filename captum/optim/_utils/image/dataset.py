import os
from typing import List, Optional, Tuple, cast

import torch

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " image dataset functions with progress bar"
    )

from captum.optim._utils.models import collect_activations


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


def attribute_spatial_position(
    target_activ: torch.Tensor,
    logit_activ: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        logit_activ: Captured activations from the FC / logit layer.
        target_activ: Captured activations from the target layer.
        position_mask (torch.Tensor, optional): If using a batch size greater than
        one, a mask is used to zero out all the non-target positions.
    Returns:
        logit_attr (torch.Tensor): A sorted list of class attributions for the target
            spatial positions.
    """

    assert target_activ.dim() == 2 or target_activ.dim() == 4
    assert logit_activ.dim() == 2

    zeros = torch.nn.Parameter(torch.zeros_like(logit_activ))
    target_zeros = target_activ * position_mask

    grad_one = torch.autograd.grad(
        outputs=[logit_activ],
        inputs=[target_activ],
        grad_outputs=[zeros],
        create_graph=True,
    )
    logit_attr = torch.autograd.grad(
        outputs=grad_one,
        inputs=[zeros],
        grad_outputs=[target_zeros],
        create_graph=True,
    )[0]
    return logit_attr


def capture_activation_samples(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    targets: List[torch.nn.Module],
    target_names: Optional[List[str]] = None,
    sample_dir: str = "",
    num_images: Optional[int] = None,
    samples_per_image: int = 1,
    input_device: torch.device = torch.device("cpu"),
    collect_attributions: bool = False,
    attr_model: Optional[torch.nn.Module] = None,
    attr_targets: Optional[List[torch.nn.Module]] = None,
    logit_target: Optional[torch.nn.Module] = None,
    show_progress: bool = False,
):
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
        collect_attributions (bool, optional): Whether or not to collect attributions
            for samples.
        attr_model (nn.Module, optional): A PyTorch model instance to use for
            calculating sample attributions.
        attr_targets (list of nn.Module, optional): A list of attribution model layers
            to collect attributions from. This should be the exact same as the targets
            parameter, except for the attribution model.
        logit_target (nn.Module, optional): The final layer in the attribution model
            that determines the classes. This parameter is only enabled if
            collect_attributions is set to True.
        show_progress (bool, optional): Whether or not to show progress.
    """

    if target_names is None:
        target_names = ["target" + str(i) + "_" for i in range(len(targets))]

    assert len(target_names) == len(targets)
    assert os.path.isdir(sample_dir)

    def random_sample(
        activations: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[List[int]]]]:
        """
        Randomly sample H & W dimensions of activations with 4 dimensions.
        """
        assert activations.dim() == 4 or activations.dim() == 2

        activation_samples: List = []
        position_list: List = []

        with torch.no_grad():
            for i in range(samples_per_image):
                sample_position_list: List = []
                for b in range(activations.size(0)):
                    if activations.dim() == 4:
                        h, w = activations.shape[2:]
                        y = torch.randint(low=1, high=h - 1, size=[1])
                        x = torch.randint(low=1, high=w - 1, size=[1])
                        activ = activations[b, :, y, x]
                        sample_position_list.append((b, y, x))
                    elif activations.dim() == 2:
                        activ = activations[b].unsqueeze(1)
                        sample_position_list.append(b)
                    activation_samples.append(activ)
                position_list.append(sample_position_list)
        return activation_samples, position_list

    def attribute_samples(
        activations: torch.Tensor,
        logit_activ: torch.Tensor,
        position_list: List[List[List[int]]],
    ) -> List[torch.Tensor]:
        """
        Collect attributions for target sample positions.
        """
        assert activations.dim() == 4 or activations.dim() == 2

        sample_attributions: List = []
        with torch.set_grad_enabled(True):
            zeros_mask = torch.zeros_like(activations)
            for sample_pos_list in position_list:
                for c in sample_pos_list:
                    if activations.dim() == 4:
                        zeros_mask[c[0], :, c[1], c[2]] = 1
                    elif activations.dim() == 2:
                        zeros_mask[c] = 1
                attr = attribute_spatial_position(
                    activations, logit_activ, position_mask=zeros_mask
                ).detach()
                sample_attributions.append(attr)
        return sample_attributions

    if collect_attributions:
        logit_target == list(model.children())[len(list(model.children())) - 1 :][
            0
        ] if logit_target is None else logit_target
        attr_targets = cast(List[torch.nn.Module], attr_targets)
        attr_targets += [cast(torch.nn.Module, logit_target)]

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
            if collect_attributions:
                with torch.set_grad_enabled(True):
                    target_activ_attr_dict = collect_activations(
                        attr_model, attr_targets, inputs
                    )
                    logit_activ = target_activ_attr_dict[logit_target]
                    del target_activ_attr_dict[logit_target]

            sample_coords = []
            for t, n in zip(target_activ_dict, target_names):
                sample_tensors, p_list = random_sample(target_activ_dict[t])
                torch.save(
                    sample_tensors,
                    os.path.join(
                        sample_dir, n + "_activations_" + str(batch_count) + ".pt"
                    ),
                )
                sample_coords.append(p_list)

            if collect_attributions:
                for t, n, s_coords in zip(
                    target_activ_attr_dict, target_names, sample_coords
                ):
                    sample_attrs = attribute_samples(
                        target_activ_attr_dict[t], logit_activ, s_coords
                    )
                    torch.save(
                        sample_attrs,
                        os.path.join(
                            sample_dir,
                            n + "_attributions_" + str(batch_count) + ".pt",
                        ),
                    )

            if show_progress:
                pbar.update(inputs.size(0))

            if num_images is not None:
                if image_count > num_images:
                    break

    if show_progress:
        pbar.close()


def consolidate_samples(
    sample_dir: str = "samples",
    sample_basename: str = "",
    dim: int = 1,
    num_files: Optional[int] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Combine samples collected from capture_activation_samples into a single tensor
    with a shape of [n_channels, n_samples].

    Args:
        sample_dir (str): The directory where activation samples where saved.
        sample_basename (str, optional): If samples from different layers are present
            in sample_dir, then you can use samples from only a specific layer by
            specifying the basename that samples of the same layer share.
        dim (int, optional): The dimension to concatinate the samples together on.
        show_progress (bool, optional): Whether or not to show progress.
    Returns:
        sample_tensor (torch.Tensor): A tensor containing all the specified sample
            tensors with a shape of [n_channels, n_samples].
    """

    assert os.path.isdir(sample_dir)

    tensor_samples = [
        os.path.join(sample_dir, name)
        for name in os.listdir(sample_dir)
        if sample_basename.lower() in name.lower()
        and os.path.isfile(os.path.join(sample_dir, name))
    ]
    assert len(tensor_samples) > 0

    if show_progress:
        total = len(tensor_samples) if num_files is None else num_files  # type: ignore
        pbar = tqdm(total=total, unit=" sample batches collected")

    samples: List[torch.Tensor] = []
    for file in tensor_samples:
        sample_batch = torch.load(file)
        for s in sample_batch:
            samples += [s.cpu()]
        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()
    return torch.cat(samples, dim)
