from typing import List, Optional, Tuple

import torch

def get_neuron_pos(
    H: int, W: int, x: Optional[int] = None, y: Optional[int] = None
) -> Tuple[int, int]:
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


def nchannels_to_rgb(x: torch.Tensor, warp: bool = True) -> torch.Tensor:
    """
    Convert an NCHW image with n channels into a 3 channel RGB image.

    Args:
        x (torch.Tensor):  Image tensor to transform into RGB image.
        warp (bool, optional):  Whether or not to make colors more distinguishable.
            Default: True
    Returns:
        *tensor* RGB image
    """

    def hue_to_rgb(angle: float) -> torch.Tensor:
        """
        Create an RGB unit vector based on a hue of the input angle.
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
            ]
        )

        idx = math.floor(angle / 60)
        d = (angle - idx * 60) / 60

        if warp:

            def adj(x: float) -> float:
                return math.sin(x * math.pi / 2)

            d = adj(d) if idx % 2 == 0 else 1 - adj(1 - d)

        vec = (1 - d) * colors[idx] + d * colors[(idx + 1) % 6]
        return vec / torch.norm(vec)

    assert x.dim() == 4

    if (x < 0).any():
        x = posneg(x.permute(0, 2, 3, 1), -1).permute(0, 3, 1, 2)

    rgb = torch.zeros(1, 3, x.size(2), x.size(3), device=x.device)
    nc = x.size(1)
    for i in range(nc):
        rgb = rgb + x[:, i][:, None, :, :]
        rgb = rgb * hue_to_rgb(360 * i / nc).to(device=x.device)[None, :, None, None]

    rgb = rgb + torch.ones(x.size(2), x.size(3))[None, None, :, :] * (
        torch.sum(x, 1)[:, None] - torch.max(x, 1)[0][:, None]
    )
    return (rgb / (1e-4 + torch.norm(rgb, dim=1, keepdim=True))) * torch.norm(
        x, dim=1, keepdim=True
    )


def tensor_heatmap(
    tensor: torch.Tensor,
    colors: List[str] = ["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"],
) -> torch.Tensor:
    """
    Create a color heatmap of an input weight tensor.
    By default red represents excitatory values,
    blue represents inhibitory values, and white represents
    no excitation or inhibition.
    """

    assert tensor.dim() == 2
    assert len(colors) == 5

    def get_color(x: str) -> torch.Tensor:
        def hex2base10(x: str) -> float:
            return int(x, 16) / 255.0

        return torch.tensor(
            [hex2base10(x[0:2]), hex2base10(x[2:4]), hex2base10(x[4:6])]
        )

    def color_scale(x: torch.Tensor) -> torch.Tensor:
        if x < 0:
            x = -x
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2]) + x * get_color(colors[1])
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[1]) + x * get_color(colors[0])
        else:
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2]) + x * get_color(colors[3])
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[3]) + x * get_color(colors[4])

    return torch.stack(
        [torch.stack([color_scale(x) for x in t]) for t in tensor]
    ).permute(2, 0, 1)
