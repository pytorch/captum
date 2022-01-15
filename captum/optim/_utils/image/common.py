import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from captum.optim._utils.reducer import posneg

try:
    from PIL import Image
except (ImportError, AssertionError):
    print("The Pillow/PIL library is required to use Captum's Optim library")


def show(
    x: torch.Tensor, figsize: Optional[Tuple[int, int]] = None, scale: float = 255.0
) -> None:
    """
    Show CHW & NCHW tensors as an image.

    Args:
        x (torch.Tensor): The tensor you want to display as an image.
        figsize (Tuple[int, int], optional): height & width to use
            for displaying the image figure.
        scale (float): Value to multiply the input tensor by so that
            it's value range is [0-255] for display.
    """

    if x.dim() not in [3, 4]:
        raise ValueError(
            f"Incompatible number of dimensions. x.dim() = {x.dim()}; should be 3 or 4."
        )
    x = torch.cat([t[0] for t in x.split(1)], dim=2) if x.dim() == 4 else x
    x = x.clone().cpu().detach().permute(1, 2, 0) * scale
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(x.numpy().astype(np.uint8))
    plt.axis("off")
    plt.show()


def save_tensor_as_image(x: torch.Tensor, filename: str, scale: float = 255.0) -> None:
    """
    Save RGB & RGBA image tensors with a shape of CHW or NCHW as images.

    Args:
        x (torch.Tensor): The tensor you want to save as an image.
        filename (str): The filename to use when saving the image.
        scale (float, optional): Value to multiply the input tensor by so that
            it's value range is [0-255] for saving.
    """

    if x.dim() not in [3, 4]:
        raise ValueError(
            f"Incompatible number of dimensions. x.dim() = {x.dim()}; should be 3 or 4."
        )
    x = x[0] if x.dim() == 4 else x
    x = x.clone().cpu().detach().permute(1, 2, 0) * scale
    colorspace = "RGB" if x.shape[2] == 3 else "RGBA"
    im = Image.fromarray(x.numpy().astype(np.uint8), colorspace)
    im.save(filename)


def get_neuron_pos(
    H: int, W: int, x: Optional[int] = None, y: Optional[int] = None
) -> Tuple[int, int]:
    """
    Args:

        H (int) The height
        W (int) The width
        x (int, optional): Optionally specify and exact x location of the neuron. If
            set to None, then the center x location will be used.
            Default: None
        y (int, optional): Optionally specify and exact y location of the neuron. If
            set to None, then the center y location will be used.
            Default: None

    Return:
        Tuple[_x, _y] (Tuple[int, int]): The x and y dimensions of the neuron.
    """
    if x is None:
        _x = W // 2
    else:
        assert x < W
        _x = x

    if y is None:
        _y = H // 2
    else:
        assert y < H
        _y = y
    return _x, _y


def _dot_cossim(
    x: torch.Tensor,
    y: torch.Tensor,
    cossim_pow: float = 0.0,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes product between dot product and cosine similarity of two tensors along
    a specified dimension.

    Args:
        x (torch.Tensor): The tensor that you wish to compute the cosine similarity
            for in relation to tensor y.
        y (torch.Tensor): The tensor that you wish to compute the cosine similarity
            for in relation to tensor x.
        cossim_pow (float, optional): The desired cosine similarity power to use.
        dim (int, optional): The target dimension for computing cosine similarity.
        eps (float, optional): If cossim_pow is greater than zero, the desired
            epsilon value to use for cosine similarity calculations.
    Returns:
        tensor (torch.Tensor): Dot cosine similarity between x and y, along the
        specified dim.
    """

    dot = torch.sum(x * y, dim)
    if cossim_pow == 0:
        return dot
    return dot * torch.clamp(torch.cosine_similarity(x, y, eps=eps), 0.1) ** cossim_pow


# Handle older versions of PyTorch
# Defined outside of function in order to support JIT
_torch_norm = torch.linalg.norm if torch.__version__ >= "1.9.0" else torch.norm


def hue_to_rgb(
    angle: float, device: torch.device = torch.device("cpu"), warp: bool = True
) -> torch.Tensor:
    """
    Create an RGB unit vector based on a hue of the input angle.
    Args:
        angle (float): The hue angle to create an RGB color for.
        device (torch.device, optional): The device to create the angle color tensor
            on.
            Default: torch.device("cpu")
        warp (bool, optional): Whether or not to make colors more distinguishable.
            Default: True
    Returns:
        color_vec (torch.Tensor): A color vector.
    """

    angle = angle - 360 * (angle // 360)
    colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7071, 0.7071],
            [0.0, 0.0, 1.0],
            [0.7071, 0.0, 0.7071],
        ],
        device=device,
    )

    idx = math.floor(angle / 60)
    d = (angle - idx * 60) / 60

    if warp:
        # Idea from: https://github.com/tensorflow/lucid/pull/193
        d = (
            math.sin(d * math.pi / 2)
            if idx % 2 == 0
            else 1 - math.sin((1 - d) * math.pi / 2)
        )

    vec = (1 - d) * colors[idx] + d * colors[(idx + 1) % 6]
    return vec / _torch_norm(vec)


def nchannels_to_rgb(
    x: torch.Tensor, warp: bool = True, eps: float = 1e-4
) -> torch.Tensor:
    """
    Convert an NCHW image with n channels into a 3 channel RGB image.
    Args:
        x (torch.Tensor):  NCHW image tensor to transform into RGB image.
        warp (bool, optional):  Whether or not to make colors more distinguishable.
            Default: True
        eps (float, optional): An optional epsilon value.
            Default: 1e-4
    Returns:
        tensor (torch.Tensor): An NCHW RGB image tensor.
    """

    assert x.dim() == 4

    if (x < 0).any():
        x = posneg(x.permute(0, 2, 3, 1), -1).permute(0, 3, 1, 2)

    rgb = torch.zeros(1, 3, x.size(2), x.size(3), device=x.device)
    num_channels = x.size(1)
    for i in range(num_channels):
        rgb_angle = hue_to_rgb(360 * i / num_channels, device=x.device, warp=warp)
        rgb = rgb + (x[:, i][:, None, :, :] * rgb_angle[None, :, None, None])

    rgb = rgb + (
        torch.ones(1, 1, x.size(2), x.size(3), device=x.device)
        * (torch.sum(x, 1) - torch.max(x, 1)[0])[:, None]
    )
    rgb = rgb / (eps + _torch_norm(rgb, dim=1, keepdim=True))
    return rgb * _torch_norm(x, dim=1, keepdim=True)


def weights_to_heatmap_2d(
    weight: torch.Tensor,
    colors: List[str] = ["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"],
) -> torch.Tensor:
    """
    Create a color heatmap of an input weight tensor.
    By default red represents excitatory values,
    blue represents inhibitory values, and white represents
    no excitation or inhibition.

    Args:
        weight (torch.Tensor):  A 2d tensor to create the heatmap from.
        colors (List of strings):  A list of strings containing color
        hex values to use for coloring the heatmap.
    Returns:
        *tensor*:  A weight heatmap.
    """

    assert weight.dim() == 2
    assert len(colors) == 5

    def get_color(x: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        def hex2base10(x: str) -> float:
            return int(x, 16) / 255.0

        return torch.tensor(
            [hex2base10(x[0:2]), hex2base10(x[2:4]), hex2base10(x[4:6])], device=device
        )

    def color_scale(x: torch.Tensor) -> torch.Tensor:
        if x < 0:
            x = -x
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2], x.device) + x * get_color(
                    colors[1], x.device
                )
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[1], x.device) + x * get_color(
                    colors[0], x.device
                )
        else:
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2], x.device) + x * get_color(
                    colors[3], x.device
                )
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[3], x.device) + x * get_color(
                    colors[4], x.device
                )

    return torch.stack(
        [torch.stack([color_scale(x) for x in t]) for t in weight]
    ).permute(2, 0, 1)
