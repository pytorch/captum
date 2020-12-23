import math
import numbers
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.optim._utils.image.common import nchannels_to_rgb
from captum.optim._utils.typing import TransformSize, TransformVal, TransformValList

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BlendAlpha(nn.Module):
    r"""Blends a 4 channel input parameterization into an RGB image.

    You can specify a fixed background, or a random one will be used by default.
    """

    def __init__(self, background: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.background = background

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        assert x.size(1) == 4
        rgb, alpha = x[:, :3, ...], x[:, 3:4, ...]
        background = (
            self.background if self.background is not None else torch.rand_like(rgb)
        )
        blended = alpha * rgb + (1 - alpha) * background
        return blended


class IgnoreAlpha(nn.Module):
    r"""Ignores a 4th channel"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        assert x.size(1) == 4
        rgb = x[:, :3, ...]
        return rgb


class ToRGB(nn.Module):
    """Transforms arbitrary channels to RGB. We use this to ensure our
    image parameteriaztion itself can be decorrelated. So this goes between
    the image parameterization and the normalization/sigmoid step.
    We offer two transforms: Karhunen-Loève (KLT) and I1I2I3.
    KLT corresponds to the empirically measured channel correlations on imagenet.
    I1I2I3 corresponds to an aproximation for natural images from Ohta et al.[0]
    [0] Y. Ohta, T. Kanade, and T. Sakai, "Color information for region segmentation,"
    Computer Graphics and Image Processing, vol. 13, no. 3, pp. 222–241, 1980
    https://www.sciencedirect.com/science/article/pii/0146664X80900477
    """

    @staticmethod
    def klt_transform() -> torch.Tensor:
        """Karhunen-Loève transform (KLT) measured on ImageNet"""
        KLT = [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        transform = torch.Tensor(KLT).float()
        transform = transform / torch.max(torch.norm(transform, dim=0))
        return transform

    @staticmethod
    def i1i2i3_transform() -> torch.Tensor:
        i1i2i3_matrix = [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 2, 0, -1 / 2],
            [-1 / 4, 1 / 2, -1 / 4],
        ]
        return torch.Tensor(i1i2i3_matrix)

    def __init__(self, transform_name: str = "klt") -> None:
        super().__init__()

        if transform_name == "klt":
            self.register_buffer("transform", ToRGB.klt_transform())
        elif transform_name == "i1i2i3":
            self.register_buffer("transform", ToRGB.i1i2i3_transform())
        else:
            raise ValueError("transform_name has to be either 'klt' or 'i1i2i3'")

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        assert x.dim() == 3 or x.dim() == 4

        # alpha channel is taken off...
        has_alpha = x.size("C") == 4
        if has_alpha:
            if x.dim() == 3:
                x, alpha_channel = x[:3], x[3:]
            elif x.dim() == 4:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.dim() == alpha_channel.dim()  # ensure we "keep_dim"

        h, w = x.size("H"), x.size("W")
        flat = x.flatten(("H", "W"), "spatials")
        if inverse:
            correct = torch.inverse(self.transform) @ flat
        else:
            correct = self.transform @ flat
        chw = correct.unflatten("spatials", (("H", h), ("W", w)))

        if x.dim() == 3:
            chw = chw.refine_names("C", ...)
        elif x.dim() == 4:
            chw = chw.refine_names("B", "C", ...)

        # ...alpha channel is concatenated on again.
        if has_alpha:
            d = 0 if x.dim() == 3 else 1
            chw = torch.cat([chw, alpha_channel], d)

        return chw


class CenterCrop(torch.nn.Module):
    """
    Center crop the specified amount of pixels from the edges.
    Arguments:
        size (int, sequence) or (int): Number of pixels to center crop away.
    """

    def __init__(self, size: TransformSize = 0) -> None:
        super(CenterCrop, self).__init__()
        if type(size) is list or type(size) is tuple:
            assert len(size) == 2, (
                "CenterCrop requires a single crop value or a tuple of (height,width)"
                + "in pixels for cropping."
            )
            self.crop_val = size
        else:
            self.crop_val = [size] * 2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert (
            input.dim() == 3 or input.dim() == 4
        ), "Input to CenterCrop must be 3D or 4D"
        if input.dim() == 4:
            h, w = input.size(2), input.size(3)
        elif input.dim() == 3:
            h, w = input.size(1), input.size(2)
        h_crop = h - self.crop_val[0]
        w_crop = w - self.crop_val[1]
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        return input[..., sh : sh + h_crop, sw : sw + w_crop]


def center_crop_shape(input: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    """
    Crop NCHW & CHW outputs by specifying the desired output shape.
    """

    assert input.dim() == 4 or input.dim() == 3
    output_size = [output_size] if not hasattr(output_size, "__iter__") else output_size
    assert len(output_size) == 1 or len(output_size) == 2
    output_size = output_size * 2 if len(output_size) == 1 else output_size

    if input.dim() == 4:
        h, w = input.size(2), input.size(3)
    if input.dim() == 3:
        h, w = input.size(1), input.size(2)

    h_crop = h - int(round((h - output_size[0]) / 2.0))
    w_crop = w - int(round((w - output_size[1]) / 2.0))

    return input[
        ..., h_crop - output_size[0] : h_crop, w_crop - output_size[1] : w_crop
    ]


def rand_select(transform_values: TransformValList) -> TransformVal:
    """
    Randomly return a value from the provided tuple or list
    """
    n = torch.randint(low=0, high=len(transform_values) - 1, size=[1]).item()
    return transform_values[n]


class RandomScale(nn.Module):
    """
    Apply random rescaling on a NCHW tensor.
    Arguments:
        scale (float, sequence): Tuple of rescaling values to randomly select from.
    """

    def __init__(self, scale: TransformValList) -> None:
        super(RandomScale, self).__init__()
        self.scale = scale

    def get_scale_mat(
        self, m: TransformVal, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        scale_mat = torch.tensor(
            [[m, 0.0, 0.0], [0.0, m, 0.0]], device=device, dtype=dtype
        )
        return scale_mat

    def scale_tensor(self, x: torch.Tensor, scale: TransformVal) -> torch.Tensor:
        scale_matrix = self.get_scale_mat(scale, x.device, x.dtype)[None, ...].repeat(
            x.shape[0], 1, 1
        )
        grid = F.affine_grid(scale_matrix, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = rand_select(self.scale)
        return self.scale_tensor(input, scale=scale)


class RandomSpatialJitter(torch.nn.Module):
    """
    Apply random spatial translations on a NCHW tensor.
    Arguments:
        translate (int):
    """

    def __init__(self, translate: int) -> None:
        super(RandomSpatialJitter, self).__init__()
        self.pad_range = 2 * translate
        self.pad = nn.ReflectionPad2d(translate)

    def translate_tensor(self, x: torch.Tensor, insets: torch.Tensor) -> torch.Tensor:
        padded = self.pad(x)
        tblr = [
            -insets[0],
            -(self.pad_range - insets[0]),
            -insets[1],
            -(self.pad_range - insets[1]),
        ]
        cropped = F.pad(padded, pad=tblr)
        assert cropped.shape == x.shape
        return cropped

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        insets = torch.randint(high=self.pad_range, size=(2,))
        return self.translate_tensor(input, insets)


class ScaleInputRange(nn.Module):
    """
    Multiplies the input by a specified multiplier for models with input ranges other
    than [0,1].
    """

    def __init__(self, multiplier: float = 1.0) -> None:
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.multiplier


class RGBToBGR(nn.Module):
    """
    Converts an NCHW RGB image tensor to BGR by switching the red and blue channels.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        assert x.size(1) == 3
        return x[:, [2, 1, 0]]


# class TransformationRobustness(nn.Module):
#     def __init__(self, jitter=False, scale=False):
#         super().__init__()
#         if jitter:
#             self.jitter = RandomSpatialJitter(4)
#         if scale:
#             self.scale = RandomScale()

#     def forward(self, x):
#         original_shape = x.shape
#         if hasattr(self, "jitter"):
#             x = self.jitter(x)
#         if hasattr(self, "scale"):
#             x = self.scale(x)
#         cropped = center_crop(x, original_shape)
#         return cropped


# class RandomHomography(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         _, _, H, W = x.shape
#         self.homography_warper = HomographyWarper(
#             height=H, width=W, padding_mode="reflection"
#         )
#         homography =
#         return self.homography_warper(x, homography)


# via https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-
# filtering-for-an-image-2d-3d-in-pytorch/12351/9
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[float, Sequence[float]],
        dim: int = 2,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class SymmetricPadding(torch.autograd.Function):
    """
    Autograd compatible symmetric padding that uses NumPy's pad function.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, padding: List[List[int]]) -> torch.Tensor:
        ctx.padding = padding
        x_device = x.device
        x = x.cpu()
        x.data = torch.as_tensor(
            np.pad(x.data.numpy(), pad_width=padding, mode="symmetric")
        )
        x = x.to(x_device)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_input = grad_output.clone()
        B, C, H, W = grad_input.size()
        b1, b2 = ctx.padding[0]
        c1, c2 = ctx.padding[1]
        h1, h2 = ctx.padding[2]
        w1, w2 = ctx.padding[3]
        grad_input = grad_input[b1 : B - b2, c1 : C - c2, h1 : H - h2, w1 : W - w2]
        return grad_input, None


class NChannelsToRGB(nn.Module):
    """
    Convert an NCHW image with n channels into a 3 channel RGB image.
    """

    def __init__(self, warp: bool = False) -> None:
        super().__init__()
        self.warp = warp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        return nchannels_to_rgb(x, self.warp)
