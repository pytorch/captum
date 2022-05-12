import math
import numbers
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.optim._utils.image.common import nchannels_to_rgb
from captum.optim._utils.typing import IntSeqOrIntType, NumSeqOrTensorOrProbDistType


class BlendAlpha(nn.Module):
    r"""Blends a 4 channel input parameterization into an RGB image.
    You can specify a fixed background, or a random one will be used by default.
    """

    def __init__(self, background: Optional[torch.Tensor] = None) -> None:
        """
        Args:

            background (tensor, optional):  An NCHW image tensor to be used as the
                Alpha channel's background.
                Default: None
        """
        super().__init__()
        self.background = background

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Blend the Alpha channel into the RGB channels.

        Args:

            x (torch.Tensor): RGBA image tensor to blend into an RGB image tensor.

        Returns:
            **blended** (torch.Tensor): RGB image tensor.
        """
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
        """
        Ignore the alpha channel.

        Args:

            x (torch.Tensor): RGBA image tensor.

        Returns:
            **rgb** (torch.Tensor): RGB image tensor without the alpha channel.
        """
        assert x.dim() == 4
        assert x.size(1) == 4
        rgb = x[:, :3, ...]
        return rgb


class ToRGB(nn.Module):
    """Transforms arbitrary channels to RGB. We use this to ensure our
    image parametrization itself can be decorrelated. So this goes between
    the image parametrization and the normalization/sigmoid step.
    We offer two precalculated transforms: Karhunen-Loève (KLT) and I1I2I3.
    KLT corresponds to the empirically measured channel correlations on imagenet.
    I1I2I3 corresponds to an approximation for natural images from Ohta et al.[0]
    [0] Y. Ohta, T. Kanade, and T. Sakai, "Color information for region segmentation,"
    Computer Graphics and Image Processing, vol. 13, no. 3, pp. 222–241, 1980
    https://www.sciencedirect.com/science/article/pii/0146664X80900477
    """

    __constants__ = ["_supports_is_scripting", "_supports_named_dims"]

    @staticmethod
    def klt_transform() -> torch.Tensor:
        """
        Karhunen-Loève transform (KLT) measured on ImageNet

        Returns:
            **transform** (torch.Tensor): A Karhunen-Loève transform (KLT) measured on
                the ImageNet dataset.
        """
        KLT = [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        transform = torch.Tensor(KLT).float()
        transform = transform / torch.max(torch.norm(transform, dim=0))
        return transform

    @staticmethod
    def i1i2i3_transform() -> torch.Tensor:
        """
        Returns:
            **transform** (torch.Tensor): An approximation of natural colors transform
                (i1i2i3).
        """
        i1i2i3_matrix = [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 2, 0, -1 / 2],
            [-1 / 4, 1 / 2, -1 / 4],
        ]
        return torch.Tensor(i1i2i3_matrix)

    def __init__(self, transform: Union[str, torch.Tensor] = "klt") -> None:
        """
        Args:

            transform (str or tensor):  Either a string for one of the precalculated
                transform matrices, or a 3x3 matrix for the 3 RGB channels of input
                tensors.
        """
        super().__init__()
        assert isinstance(transform, str) or torch.is_tensor(transform)
        if torch.is_tensor(transform):
            transform = cast(torch.Tensor, transform)
            assert list(transform.shape) == [3, 3]
            self.register_buffer("transform", transform)
        elif transform == "klt":
            self.register_buffer("transform", ToRGB.klt_transform())
        elif transform == "i1i2i3":
            self.register_buffer("transform", ToRGB.i1i2i3_transform())
        else:
            raise ValueError(
                "transform has to be either 'klt', 'i1i2i3'," + " or a matrix tensor."
            )
        # Check & store whether or not we can use torch.jit.is_scripting()
        self._supports_is_scripting = torch.__version__ >= "1.6.0"
        self._supports_named_dims = torch.__version__ >= "1.3.0"

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Args:

            x (torch.tensor):  A CHW or NCHW RGB or RGBA image tensor.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default: False.

        Returns:
            chw (torch.tensor):  A tensor with it's colors recorrelated or
                decorrelated.
        """

        assert x.dim() == 3 or x.dim() == 4
        assert x.shape[-3] >= 3
        assert (
            x.names == ("C", "H", "W")
            if x.dim() == 3
            else x.names == ("B", "C", "H", "W")
        )

        # alpha channel is taken off...
        has_alpha = x.size("C") >= 4
        if has_alpha:
            if x.dim() == 3:
                x, alpha_channel = x[:3], x[3:]
            elif x.dim() == 4:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.dim() == alpha_channel.dim()  # ensure we "keep_dim"

        h, w = x.size("H"), x.size("W")
        flat = x.flatten(("H", "W"), "spatials")
        if inverse:
            correct = torch.inverse(self.transform.to(x.device)) @ flat
        else:
            correct = self.transform.to(x.device) @ flat
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

    def _forward_without_named_dims(
        self, x: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """
        JIT compatible forward function for ToRGB.

        Args:

            x (torch.tensor):  A CHW pr NCHW RGB or RGBA image tensor.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default: False.

        Returns:
            chw (torch.tensor):  A tensor with it's colors recorrelated or
                decorrelated.
        """

        assert x.dim() == 4 or x.dim() == 3
        assert x.shape[-3] >= 3

        # alpha channel is taken off...
        has_alpha = x.shape[-3] >= 4
        if has_alpha:
            if x.dim() == 3:
                x, alpha_channel = x[:3], x[3:]
            else:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.dim() == alpha_channel.dim()  # ensure we "keep_dim"
        else:
            # JIT requires a placeholder
            alpha_channel = torch.tensor([0])

        c_dim = 1 if x.dim() == 4 else 0
        h, w = x.shape[c_dim + 1 :]
        flat = x.reshape(list(x.shape[: c_dim + 1]) + [h * w])

        if inverse:
            correct = torch.inverse(self.transform.to(x.device, x.dtype)) @ flat
        else:
            correct = self.transform.to(x.device, x.dtype) @ flat
        chw = correct.reshape(x.shape)

        # ...alpha channel is concatenated on again.
        if has_alpha:
            d = 0 if x.dim() == 3 else 1
            chw = torch.cat([chw, alpha_channel], d)

        return chw

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        JIT does not yet support named dimensions.

        Args:

            x (torch.tensor):  A CHW or NCHW RGB or RGBA image tensor.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default: False.

        Returns:
            chw (torch.tensor):  A tensor with it's colors recorrelated or
                decorrelated.
        """
        if self._supports_is_scripting:
            if torch.jit.is_scripting():
                return self._forward_without_named_dims(x, inverse)
        if self._supports_named_dims:
            if list(x.names) in [[None] * 3, [None] * 4]:
                return self._forward_without_named_dims(x, inverse)
        return self._forward(x, inverse)


class CenterCrop(torch.nn.Module):
    """
    Center crop a specified amount from a tensor. If input are smaller than the
    specified crop size, padding will be applied.
    """

    __constants__ = [
        "size",
        "pixels_from_edges",
        "offset_left",
        "padding_mode",
        "padding_value",
    ]

    def __init__(
        self,
        size: IntSeqOrIntType = 0,
        pixels_from_edges: bool = False,
        offset_left: bool = False,
        padding_mode: str = "constant",
        padding_value: float = 0.0,
    ) -> None:
        """
        Args:

            size (int, sequence, int): Number of pixels to center crop away.
                pixels_from_edges (bool, optional): Whether to treat crop size
                values as the number of pixels from the tensor's edge, or an
                exact shape in the center.
            pixels_from_edges (bool, optional): Whether to treat crop size
                values as the number of pixels from the tensor's edge, or an
                exact shape in the center.
                Default: False
            offset_left (bool, optional): If the cropped away sides are not
                equal in size, offset center by +1 to the left and/or top.
                This parameter is only valid when `pixels_from_edges` is False.
                Default: False
            padding_mode (optional, str): One of "constant", "reflect", "replicate"
                or "circular". This parameter is only used if the crop size is larger
                than the image size.
                Default: "constant"
            padding_value (float, optional): fill value for "constant" padding. This
                parameter is only used if the crop size is larger than the image size.
                Default: 0.0
        """
        super().__init__()
        if not hasattr(size, "__iter__"):
            size = [int(size), int(size)]
        elif isinstance(size, (tuple, list)):
            if len(size) == 1:
                size = list((size[0], size[0]))
            elif len(size) == 2:
                size = list(size)
            else:
                raise ValueError("Crop size length of {} too large".format(len(size)))
        else:
            raise ValueError("Unsupported crop size value {}".format(size))
        assert len(size) == 2
        self.size = cast(List[int], size)
        self.pixels_from_edges = pixels_from_edges
        self.offset_left = offset_left
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    @torch.jit.ignore
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Center crop an input.

        Args:

            input (torch.Tensor): Input to center crop.

        Returns:
            **tensor** (torch.Tensor): A center cropped *tensor*.
        """

        return center_crop(
            input,
            self.size,
            self.pixels_from_edges,
            self.offset_left,
            self.padding_mode,
            self.padding_value,
        )


def center_crop(
    input: torch.Tensor,
    size: Union[int, List[int]],
    pixels_from_edges: bool = False,
    offset_left: bool = False,
    padding_mode: str = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Center crop a specified amount from a tensor. If input are smaller than the
    specified crop size, padding will be applied.

    Args:

        input (tensor):  A CHW or NCHW image tensor to center crop.
        size (int, sequence, int): Number of pixels to center crop away.
        pixels_from_edges (bool, optional): Whether to treat crop size
            values as the number of pixels from the tensor's edge, or an
            exact shape in the center.
            Default: False
        offset_left (bool, optional): If the cropped away sides are not
            equal in size, offset center by +1 to the left and/or top.
            This parameter is only valid when `pixels_from_edges` is False.
            Default: False
        padding_mode (optional, str): One of "constant", "reflect", "replicate" or
            "circular". This parameter is only used if the crop size is larger than
            the image size.
            Default: "constant"
        padding_value (float, optional): fill value for "constant" padding. This
            parameter is only used if the crop size is larger than the image size.
            Default: 0.0

    Returns:
        **tensor**:  A center cropped *tensor*.
    """

    assert input.dim() == 3 or input.dim() == 4
    if isinstance(size, int):
        size = [int(size), int(size)]
    elif isinstance(size, (tuple, list)):
        if len(size) == 1:
            size = [size[0], size[0]]
        elif len(size) == 2:
            size = list(size)
        else:
            raise ValueError("Crop size length of {} too large".format(len(size)))
    else:
        raise ValueError("Unsupported crop size value {}".format(size))
    assert len(size) == 2

    if input.dim() == 4:
        h, w = input.shape[2:]
    elif input.dim() == 3:
        h, w = input.shape[1:]
    else:
        raise ValueError("Input has too many dimensions: {}".format(input.dim()))

    if pixels_from_edges:
        h_crop = h - size[0]
        w_crop = w - size[1]
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        x = input[..., sh : sh + h_crop, sw : sw + w_crop]
    else:
        h_crop = h - int(math.ceil((h - size[0]) / 2.0)) if h > size[0] else size[0]
        w_crop = w - int(math.ceil((w - size[1]) / 2.0)) if w > size[1] else size[1]

        if h % 2 == 0 and size[0] % 2 != 0 or h % 2 != 0 and size[0] % 2 == 0:
            h_crop = h_crop + 1 if offset_left else h_crop
        if w % 2 == 0 and size[1] % 2 != 0 or w % 2 != 0 and size[1] % 2 == 0:
            w_crop = w_crop + 1 if offset_left else w_crop

        if size[0] > h or size[1] > w:
            # Padding functionality like Torchvision's center crop
            padding = [
                math.ceil((size[1] - w) / 2) if size[1] > w else 0,
                math.ceil((size[0] - h) / 2) if size[0] > h else 0,
                (size[1] - w + 1) // 2 if size[1] > w else 0,
                (size[0] - h + 1) // 2 if size[0] > h else 0,
            ]
            input = F.pad(input, padding, mode=padding_mode, value=padding_value)

        x = input[..., h_crop - size[0] : h_crop, w_crop - size[1] : w_crop]
    return x


class RandomScale(nn.Module):
    """
    Apply random rescaling on a NCHW tensor using the F.interpolate function.
    """

    __constants__ = [
        "scale",
        "mode",
        "align_corners",
        "_has_align_corners",
        "recompute_scale_factor",
        "_has_recompute_scale_factor",
        "_is_distribution",
    ]

    def __init__(
        self,
        scale: NumSeqOrTensorOrProbDistType,
        mode: str = "bilinear",
        align_corners: Optional[bool] = False,
        recompute_scale_factor: bool = False,
    ) -> None:
        """
        Args:

            scale (float, sequence, or torch.distribution): Sequence of rescaling
                values to randomly select from, or a torch.distributions instance.
            mode (str, optional): Interpolation mode to use. See documentation of
                F.interpolate for more details. One of; "bilinear", "nearest", "area",
                or "bicubic".
                Default: "bilinear"
            align_corners (bool, optional): Whether or not to align corners. See
                documentation of F.interpolate for more details.
                Default: False
            recompute_scale_factor (bool, optional): Whether or not to recompute the
                scale factor See documentation of F.interpolate for more details.
                Default: False
        """
        super().__init__()
        assert mode not in ["linear", "trilinear"]
        if isinstance(scale, torch.distributions.distribution.Distribution):
            # Distributions are not supported by TorchScript / JIT yet
            assert scale.batch_shape == torch.Size([])
            self.scale_distribution = scale
            self._is_distribution = True
            self.scale = []
        else:
            assert hasattr(scale, "__iter__")
            if torch.is_tensor(scale):
                assert cast(torch.Tensor, scale).dim() == 1
                scale = scale.tolist()
            assert len(scale) > 0
            self.scale = [float(s) for s in scale]
            self._is_distribution = False
        self.mode = mode
        self.align_corners = align_corners if mode not in ["nearest", "area"] else None
        self.recompute_scale_factor = recompute_scale_factor
        self._has_align_corners = torch.__version__ >= "1.3.0"
        self._has_recompute_scale_factor = torch.__version__ >= "1.6.0"

    def _scale_tensor(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale an NCHW image tensor based on a specified scale value.

        Args:

            x (torch.Tensor): The NCHW image tensor to scale.
            scale (float): The amount to scale the NCHW image by.

        Returns:
            **x** (torch.Tensor): A scaled NCHW image tensor.
        """
        if self._has_align_corners:
            if self._has_recompute_scale_factor:
                x = F.interpolate(
                    x,
                    scale_factor=scale,
                    mode=self.mode,
                    align_corners=self.align_corners,
                    recompute_scale_factor=self.recompute_scale_factor,
                )
            else:
                x = F.interpolate(
                    x,
                    scale_factor=scale,
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
        else:
            x = F.interpolate(x, scale_factor=scale, mode=self.mode)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly scale an NCHW image tensor.

        Args:

            x (torch.Tensor): NCHW image tensor to randomly scale.

        Returns:
            **x** (torch.Tensor): A randomly scaled NCHW image *tensor*.
        """
        assert x.dim() == 4
        if self._is_distribution:
            scale = float(self.scale_distribution.sample().item())
        else:
            n = int(
                torch.randint(
                    low=0,
                    high=len(self.scale),
                    size=[1],
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=x.device,
                ).item()
            )
            scale = self.scale[n]
        return self._scale_tensor(x, scale=scale)


class RandomScaleAffine(nn.Module):
    """
    Apply random rescaling on a NCHW tensor.

    This random scaling transform utilizes F.affine_grid & F.grid_sample, and as a
    result has two key differences to the default RandomScale transforms This
    transform either shrinks an image while adding a background, or center crops image
    and then resizes it to a larger size. This means that the output image shape is the
    same shape as the input image.

    In constrast to RandomScaleAffine, the default RandomScale transform simply resizes
    the input image using F.interpolate.
    """

    __constants__ = [
        "scale",
        "mode",
        "padding_mode",
        "align_corners",
        "_has_align_corners",
        "_is_distribution",
    ]

    def __init__(
        self,
        scale: NumSeqOrTensorOrProbDistType,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> None:
        """
        Args:

            scale (float, sequence, or torch.distribution): Sequence of rescaling
                values to randomly select from, or a torch.distributions instance.
            mode (str, optional): Interpolation mode to use. See documentation of
                F.grid_sample for more details. One of; "bilinear", "nearest", or
                "bicubic".
                Default: "bilinear"
            padding_mode (str, optional): Padding mode for values that fall outside of
                the grid. See documentation of F.grid_sample for more details. One of;
                "zeros", "border", or "reflection".
                Default: "zeros"
            align_corners (bool, optional): Whether or not to align corners. See
                documentation of F.affine_grid & F.grid_sample for more details.
                Default: False
        """
        super().__init__()
        if isinstance(scale, torch.distributions.distribution.Distribution):
            # Distributions are not supported by TorchScript / JIT yet
            assert scale.batch_shape == torch.Size([])
            self.scale_distribution = scale
            self._is_distribution = True
            self.scale = []
        else:
            assert hasattr(scale, "__iter__")
            if torch.is_tensor(scale):
                assert cast(torch.Tensor, scale).dim() == 1
                scale = scale.tolist()
            assert len(scale) > 0
            self.scale = [float(s) for s in scale]
            self._is_distribution = False
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self._has_align_corners = torch.__version__ >= "1.3.0"

    def _get_scale_mat(
        self,
        m: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create a scale matrix tensor.

        Args:

            m (float): The scale value to use.

        Returns:
            **scale_mat** (torch.Tensor): A scale matrix.
        """
        scale_mat = torch.tensor(
            [[m, 0.0, 0.0], [0.0, m, 0.0]], device=device, dtype=dtype
        )
        return scale_mat

    def _scale_tensor(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale an NCHW image tensor based on a specified scale value.

        Args:

            x (torch.Tensor): The NCHW image tensor to scale.
            scale (float): The amount to scale the NCHW image by.

        Returns:
            **x** (torch.Tensor): A scaled NCHW image tensor.
        """
        scale_matrix = self._get_scale_mat(scale, x.device, x.dtype)[None, ...].repeat(
            x.shape[0], 1, 1
        )
        if self._has_align_corners:
            # Pass align_corners explicitly for torch >= 1.3.0
            grid = F.affine_grid(
                scale_matrix, x.size(), align_corners=self.align_corners
            )
            x = F.grid_sample(
                x,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
        else:
            grid = F.affine_grid(scale_matrix, x.size())
            x = F.grid_sample(x, grid, mode=self.mode, padding_mode=self.padding_mode)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly scale an NCHW image tensor.

        Args:

            x (torch.Tensor): NCHW image tensor to randomly scale.

        Returns:
            **x** (torch.Tensor): A randomly scaled NCHW image *tensor*.
        """
        assert x.dim() == 4
        if self._is_distribution:
            scale = float(self.scale_distribution.sample().item())
        else:
            n = int(
                torch.randint(
                    low=0,
                    high=len(self.scale),
                    size=[1],
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=x.device,
                ).item()
            )
            scale = self.scale[n]
        return self._scale_tensor(x, scale=scale)


class RandomSpatialJitter(torch.nn.Module):
    """
    Apply random spatial translations on a NCHW tensor.
    """

    __constants__ = ["pad_range"]

    def __init__(self, translate: int) -> None:
        """
        Args:

            translate (int): The max horizontal and vertical translation to use.
        """
        super().__init__()
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
        cropped = F.pad(padded, pad=[int(n) for n in tblr])
        assert cropped.shape == x.shape
        return cropped

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Randomly translate an input tensor's height and width dimensions.

        Args:

            input (torch.Tensor): Input to randomly translate.

        Returns:
            **tensor** (torch.Tensor): A randomly translated *tensor*.
        """
        insets = torch.randint(
            high=self.pad_range,
            size=(2,),
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )
        return self.translate_tensor(input, insets)


class RandomRotation(nn.Module):
    """
    Apply random rotation transforms on a NCHW tensor, using a sequence of degrees or
    torch.distributions instance.
    """

    __constants__ = [
        "degrees",
        "mode",
        "padding_mode",
        "align_corners",
        "_has_align_corners",
        "_is_distribution",
    ]

    def __init__(
        self,
        degrees: NumSeqOrTensorOrProbDistType,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> None:
        """
        Args:

            degrees (float, sequence, or torch.distribution): Tuple of degrees values
                to randomly select from, or a torch.distributions instance.
            mode (str, optional): Interpolation mode to use. See documentation of
                F.grid_sample for more details. One of; "bilinear", "nearest", or
                "bicubic".
                Default: "bilinear"
            padding_mode (str, optional): Padding mode for values that fall outside of
                the grid. See documentation of F.grid_sample for more details. One of;
                "zeros", "border", or "reflection".
                Default: "zeros"
            align_corners (bool, optional): Whether or not to align corners. See
                documentation of F.affine_grid & F.grid_sample for more details.
                Default: False
        """
        super().__init__()
        if isinstance(degrees, torch.distributions.distribution.Distribution):
            # Distributions are not supported by TorchScript / JIT yet
            assert degrees.batch_shape == torch.Size([])
            self.degrees_distribution = degrees
            self._is_distribution = True
            self.degrees = []
        else:
            assert hasattr(degrees, "__iter__")
            if torch.is_tensor(degrees):
                assert cast(torch.Tensor, degrees).dim() == 1
                degrees = degrees.tolist()
            assert len(degrees) > 0
            self.degrees = [float(d) for d in degrees]
            self._is_distribution = False

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self._has_align_corners = torch.__version__ >= "1.3.0"

    def _get_rot_mat(
        self,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create a rotation matrix tensor.

        Args:

            theta (float): The rotation value in degrees.

        Returns:
            **rot_mat** (torch.Tensor): A rotation matrix.
        """
        theta = theta * math.pi / 180.0
        rot_mat = torch.tensor(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        return rot_mat

    def _rotate_tensor(self, x: torch.Tensor, theta: float) -> torch.Tensor:
        """
        Rotate an NCHW image tensor based on a specified degree value.

        Args:

            x (torch.Tensor): The NCHW image tensor to rotate.
            theta (float): The amount to rotate the NCHW image, in degrees.

        Returns:
            **x** (torch.Tensor): A rotated NCHW image tensor.
        """
        rot_matrix = self._get_rot_mat(theta, x.device, x.dtype)[None, ...].repeat(
            x.shape[0], 1, 1
        )
        if self._has_align_corners:
            # Pass align_corners explicitly for torch >= 1.3.0
            grid = F.affine_grid(rot_matrix, x.size(), align_corners=self.align_corners)
            x = F.grid_sample(
                x,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
        else:
            grid = F.affine_grid(rot_matrix, x.size())
            x = F.grid_sample(x, grid, mode=self.mode, padding_mode=self.padding_mode)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly rotate an NCHW image tensor.

        Args:

            x (torch.Tensor): NCHW image tensor to randomly rotate.

        Returns:
            **x** (torch.Tensor): A randomly rotated NCHW image *tensor*.
        """
        assert x.dim() == 4
        if self._is_distribution:
            rotate_angle = float(self.degrees_distribution.sample().item())
        else:
            n = int(
                torch.randint(
                    low=0,
                    high=len(self.degrees),
                    size=[1],
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=x.device,
                ).item()
            )
            rotate_angle = self.degrees[n]
        return self._rotate_tensor(x, rotate_angle)


class ScaleInputRange(nn.Module):
    """
    Multiplies the input by a specified multiplier for models with input ranges other
    than [0,1].
    """

    __constants__ = ["multiplier"]

    def __init__(self, multiplier: float = 1.0) -> None:
        """
        Args:

            multiplier (float, optional):  A float value used to scale the input.
        """
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale an input tensor's values.

        Args:

            x (torch.Tensor): Input to scale values of.

        Returns:
            **tensor** (torch.Tensor): tensor with it's values scaled.
        """
        return x * self.multiplier


class RGBToBGR(nn.Module):
    """
    Converts an NCHW RGB image tensor to BGR by switching the red and blue channels.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform RGB to BGR conversion on an input

        Args:

            x (torch.Tensor): RGB image tensor to convert to BGR.

        Returns:
            **BGR tensor** (torch.Tensor): A BGR tensor.
        """
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
    """

    __constants__ = ["groups"]

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[float, Sequence[float]],
        dim: int = 2,
    ) -> None:
        """
        Args:

            channels (int, sequence): Number of channels of the input tensors. Output
                will have this number of channels as well.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigma (float, sequence): Standard deviation of the gaussian kernel.
            dim (int, optional): The number of dimensions of the data.
                Default value is 2 (spatial).
        """
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

        Args:

            input (torch.Tensor): Input to apply gaussian filter on.

        Returns:
            **filtered** (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class SymmetricPadding(torch.autograd.Function):
    """
    Autograd compatible symmetric padding that uses NumPy's pad function.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.Function, x: torch.Tensor, padding: List[List[int]]
    ) -> torch.Tensor:
        """
        Apply NumPy symmetric padding to an input tensor while preserving the gradient.

        Args:

            x (torch.Tensor): Input to apply symmetric padding on.

        Returns:
            **tensor** (torch.Tensor): Padded tensor.
        """
        ctx.padding = padding
        x_device = x.device
        x = x.cpu()
        x.data = torch.as_tensor(
            np.pad(x.data.numpy(), pad_width=padding, mode="symmetric")
        )
        x = x.to(x_device)
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Crop away symmetric padding.

        Args:

            grad_output (torch.Tensor): Input to remove symmetric padding from.

        Returns:
            **grad_input** (torch.Tensor): Unpadded tensor.
        """
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

    __constants__ = ["warp"]

    def __init__(self, warp: bool = False) -> None:
        """
        Args:

            warp (bool, optional): Whether or not to make the resulting RGB colors more
                distict from each other. Default is set to False.
        """
        super().__init__()
        self.warp = warp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce any number of channels down to 3.

        Args:

            x (torch.Tensor): Input to reduce channel dimensions on.

        Returns:
            **3 channel RGB tensor** (torch.Tensor): RGB image tensor.
        """
        assert x.dim() == 4
        return nchannels_to_rgb(x, self.warp)


class RandomCrop(nn.Module):
    """
    Randomly crop out a specific size from an NCHW image tensor.
    """

    __constants__ = ["crop_size"]

    def __init__(
        self,
        crop_size: IntSeqOrIntType,
    ) -> None:
        """
        Args:

            crop_size (int, sequence, int): The desired cropped output size.
        """
        super().__init__()
        crop_size = [crop_size] * 2 if not hasattr(crop_size, "__iter__") else crop_size
        crop_size = list(crop_size) * 2 if len(crop_size) == 1 else crop_size
        crop_size = cast(Union[List[int], Tuple[int, int]], crop_size)
        assert len(crop_size) == 2
        self.crop_size = crop_size

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        """
        Center crop an NCHW image tensor based on self.crop_size.

        Args:

            x (torch.Tensor): The NCHW image tensor to center crop.

        Returns
            x (torch.Tensor): The center cropped NCHW image tensor.
        """
        h, w = x.shape[2:]
        h_crop = h - int(math.ceil((h - self.crop_size[0]) / 2.0))
        w_crop = w - int(math.ceil((w - self.crop_size[1]) / 2.0))
        return x[
            ...,
            h_crop - self.crop_size[0] : h_crop,
            w_crop - self.crop_size[1] : w_crop,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        hs = int(math.ceil((x.shape[2] - self.crop_size[0]) / 2.0))
        ws = int(math.ceil((x.shape[3] - self.crop_size[1]) / 2.0))
        shifts = [
            torch.randint(
                low=-hs,
                high=hs,
                size=[1],
                dtype=torch.int64,
                layout=torch.strided,
                device=x.device,
            ),
            torch.randint(
                low=-ws,
                high=ws,
                size=[1],
                dtype=torch.int64,
                layout=torch.strided,
                device=x.device,
            ),
        ]
        x = torch.roll(x, [int(s) for s in shifts], dims=(2, 3))
        return self._center_crop(x)


class TransformationRobustness(nn.Module):
    """
    This transform combines the standard transforms together for ease of use.

    Multiple jitter transforms can be used to create roughly gaussian distribution
    of jitter.

    Outputs can be optionally cropped or padded so that they have the same shape as
    inputs.
    """

    __constants__ = ["crop_or_pad_output"]

    def __init__(
        self,
        padding_transform: Optional[nn.Module] = nn.ConstantPad2d(2, value=0.5),
        translate: Optional[Union[int, List[int]]] = [4] * 10,
        scale: Optional[NumSeqOrTensorOrProbDistType] = [
            0.995 ** n for n in range(-5, 80)
        ]
        + [0.998 ** n for n in 2 * list(range(20, 40))],
        degrees: Optional[NumSeqOrTensorOrProbDistType] = list(range(-20, 20))
        + list(range(-10, 10))
        + list(range(-5, 5))
        + 5 * [0],
        final_translate: Optional[int] = 2,
        crop_or_pad_output: bool = False,
    ) -> None:
        """
        Args:

            padding_transform (nn.Module, optional): A padding module instance. No
                padding will be applied before transforms if set to None.
                Default: nn.ConstantPad2d(2, value=0.5)
            translate (int or list of int, optional): The max horizontal and vertical
                 translation to use for each jitter transform.
                 Default: [4] * 10
            scale (float, sequence, or torch.distribution, optional): Sequence of
                rescaling values to randomly select from, or a torch.distributions
                instance. If set to None, no rescaling transform will be used.
                Default: A set of optimal values.
            degrees (float, sequence, or torch.distribution, optional): Sequence of
                degrees to randomly select from, or a torch.distributions
                instance. If set to None, no rotation transform will be used.
                Default: A set of optimal values.
            final_translate (int, optional): The max horizontal and vertical
                 translation to use for the final jitter transform on fractional
                 pixels.
                 Default: 2
            crop_or_pad_output (bool, optional): Whether or not to crop or pad the
                transformed output so that it is the same shape as the input.
                Default: False
        """
        super().__init__()
        self.padding_transform = padding_transform
        if translate is not None:
            jitter_transforms = []
            if hasattr(translate, "__iter__"):
                jitter_transforms = []
                for t in translate:
                    jitter_transforms.append(RandomSpatialJitter(t))
                self.jitter_transforms = nn.Sequential(*jitter_transforms)
            else:
                self.jitter_transforms = RandomSpatialJitter(translate)
        else:
            self.jitter_transforms = translate
        self.random_scale = None if scale is None else RandomScale(scale)
        self.random_rotation = None if degrees is None else RandomRotation(degrees)
        self.final_jitter = (
            None if final_translate is None else RandomSpatialJitter(final_translate)
        )
        self.crop_or_pad_output = crop_or_pad_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        crop_size = x.shape[2:]

        # Apply padding if enabled
        if self.padding_transform is not None:
            x = self.padding_transform(x)

        # Jitter real pixels
        if self.jitter_transforms is not None:
            x = self.jitter_transforms(x)

        # Apply Random Scaling, turning real pixels into
        # fractional values of real pixels
        if self.random_scale is not None:
            x = self.random_scale(x)

        # Apply Random Rotation
        if self.random_rotation is not None:
            x = self.random_rotation(x)

        # Jitter fractional pixels if random_scale is not None
        if self.final_jitter is not None:
            x = self.final_jitter(x)

        # Ensure the output is the same shape as the input
        if self.crop_or_pad_output:
            x = center_crop(x, size=crop_size)
            assert x.shape[2:] == crop_size
        return x


__all__ = [
    "BlendAlpha",
    "IgnoreAlpha",
    "ToRGB",
    "CenterCrop",
    "center_crop",
    "RandomScale",
    "RandomSpatialJitter",
    "RandomRotation",
    "ScaleInputRange",
    "RGBToBGR",
    "GaussianSmoothing",
    "SymmetricPadding",
    "NChannelsToRGB",
    "RandomCrop",
    "TransformationRobustness",
]
