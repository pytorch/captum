import logging
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BlendAlpha(nn.Module):
    r"""Blends a 4 channel input parameterization into an RGB image.

    You can specify a fixed background, or a random one will be used by default.
    """

    def __init__(self, background: torch.Tensor = None):
        super().__init__()
        self.background = background

    def forward(self, x):
        assert x.size(1) == 4
        rgb, alpha = x[:, :3, ...], x[:, 3:4, ...]
        background = self.background or torch.rand_like(rgb)
        blended = alpha * rgb + (1 - alpha) * background
        return blended


class IgnoreAlpha(nn.Module):
    r"""Ignores a 4th channel"""

    def forward(self, x):
        assert x.size(1) == 4
        rgb = x[:, :3, ...]
        return rgb


def center_crop(input: torch.Tensor, output_size) -> torch.Tensor:
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    if len(output_size) == 4:  # assume NCHW
        output_size = output_size[2:]

    assert len(output_size) == 2 and len(input.shape) == 4

    image_width, image_height = input.shape[2:]
    height, width = output_size
    top = int(round((image_height - height) / 2.0))
    left = int(round((image_width - width) / 2.0))

    return F.pad(
        input, [top, height - image_height - top, left, width - image_width - left]
    )


def rand_select(transform_values):
    """
    Randomly return a value from the provided tuple or list
    """
    n = torch.randint(low=0, high=len(transform_values) - 1, size=[1]).item()
    return transform_values[n]


# class RandomSpatialJitter(nn.Module):
#     def __init__(self, max_distance):
#         super().__init__()

#         self.pad_range = 2 * max_distance
#         self.pad = nn.ReflectionPad2d(max_distance)

#     def forward(self, x):
#         padded = self.pad(x)
#         insets = torch.randint(high=self.pad_range, size=(2,))
#         tblr = [
#             -insets[0],
#             -(self.pad_range - insets[0]),
#             -insets[1],
#             -(self.pad_range - insets[1]),
#         ]
#         cropped = F.pad(padded, pad=tblr)
#         assert cropped.shape == x.shape
#         return cropped


class RandomScale(nn.Module):
    """
    Apply random rescaling on a NCHW tensor.
    Arguments:
        scale (float, sequence): Tuple of rescaling values to randomly select from.
    """

    def __init__(self, scale):
        super(RandomScale, self).__init__()
        self.scale = scale

    def rescale_tensor(self, input, scale):
        return torch.nn.functional.interpolate(
            input, scale_factor=scale, mode="bilinear"
        )

    def forward(self, input):
        scale = rand_select(self.scale)
        return self.rescale_tensor(input, scale=scale)


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


class RandomAffine(nn.Module):
    """
    Apply random affine transforms on a NCHW tensor.
    Arguments:
        rotate (float, sequence): Tuple of degrees to randomly select from.
        scale (float, sequence): Tuple of scale factors to randomly select from.
        shear (float, sequence): Tuple of shear values to randomly select from.
            Optionally provide a tuple that contains a tuple for the x translation
            and a tuple for y translations.
        translate (int, sequence): Tuple of values to randomly select from.
            Optionally provide a tuple that contains a tuple for the x shear values
            and a tuple for y shear values.
    """

    def __init__(self, rotate=None, scale=None, shear=None, translate=None):
        super().__init__()
        self.rotate = rotate
        self.scale = scale
        self.shear = shear if shear is None or len(shear) == 2 else [shear] * 2
        self.translate = (
            translate if translate is None or len(translate) == 2 else [translate] * 2
        )

    def get_rot_mat(self, theta, device, dtype) -> torch.Tensor:
        theta = torch.tensor(theta, device=device, dtype=dtype)
        rot_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
            ],
            device=device,
            dtype=dtype,
        )
        return rot_mat

    def rotate_tensor(self, x: torch.Tensor, theta) -> torch.Tensor:
        theta = theta * 3.141592653589793 / 180
        rot_matrix = self.get_rot_mat(theta, x.device, x.dtype)[None, ...].repeat(
            x.shape[0], 1, 1
        )
        grid = F.affine_grid(rot_matrix, x.size())
        x = F.grid_sample(x, grid)
        return x

    def get_scale_mat(self, m, device, dtype) -> torch.Tensor:
        scale_mat = torch.tensor([[m, 0.0, 0.0], [0.0, m, 0.0]])
        return scale_mat

    def scale_tensor(self, x: torch.Tensor, scale) -> torch.Tensor:
        scale_matrix = self.get_scale_mat(scale, x.device, x.dtype)[None, ...].repeat(
            x.shape[0], 1, 1
        )
        grid = F.affine_grid(scale_matrix, x.size())
        x = F.grid_sample(x, grid)
        return x

    def get_shear_mat(self, theta, ax: int, device, dtype) -> torch.Tensor:
        m = 1 / torch.tan(torch.tensor(theta, device=device, dtype=dtype))
        if ax == 0:
            shear_mat = torch.tensor([[1, m, 0], [0, 1, 0]])
        else:
            shear_mat = torch.tensor([[1, 0, 0], [m, 1, 0]])
        return shear_mat

    def shear_tensor(self, x: torch.Tensor, shear_vals) -> torch.Tensor:
        if shear_vals[0] > 0:
            shear_matrix = self.get_shear_mat(shear_vals[0], 0, x.device, x.dtype)[
                None, ...
            ].repeat(x.shape[0], 1, 1)
            grid = F.affine_grid(shear_matrix, x.size())
            x = F.grid_sample(x, grid)
        if shear_vals[1] > 0:
            shear_matrix = self.get_shear_mat(shear_vals[1], 1, x.device, x.dtype)[
                None, ...
            ].repeat(x.shape[0], 1, 1)
            grid = F.affine_grid(shear_matrix, x.size())
            x = F.grid_sample(x, grid)
        return x

    def translate_tensor(
        self, x: torch.Tensor, translation_x: int, translation_y: int
    ) -> torch.Tensor:
        x = torch.roll(x, shifts=translation_x, dims=2)
        x = torch.roll(x, shifts=translation_y, dims=3)
        return x

    def forward(self, x):
        if self.rotate is not None:
            rotate_angle = rand_select(self.rotate)
            logging.info(f"Rotate: {rotate_angle}")
            x = self.rotate(x, rotate_angle)
        if self.scale is not None:
            scale_factor = rand_select(self.scale)
            logging.info(f"Scale: {scale_factor}")
            x = self.scale_tensor(x, scale_factor)
        if self.shear is not None:
            shear_values = (rand_select(self.shear[0]), rand_select(self.shear[1]))
            logging.info(f"Shear: {shear_values}")
            x = self.shear_tensor(x, shear_values)
        if self.translate is not None:
            translations = (
                rand_select(self.translate[0]),
                rand_select(self.translate[1]),
            )
            logging.info(f"Translate: {translations}")
            x = self.translate_tensor(x, *translations)
        return x


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

    def __init__(self, channels, kernel_size, sigma, dim=2):
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

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class Normalize(nn.Module):
    """
    Apply random affine transforms on a NCHW tensor.
    Arguments:
        mean (float, sequence): Tuple of mean values to use for
            input normalization.
        std (float, sequence): Tuple of standard deviation to use for
            input normalization.
        multiplier (int): The high end of the selected model's input range.
    Returns:

    """

    def __init__(self, mean, std=[1, 1, 1], multiplier=1):
        super().__init__()
        self.mean = torch.as_tensor(mean).view(3, 1, 1).to(device)
        self.std = torch.as_tensor(std).view(3, 1, 1).to(device)
        self.multiplier = multiplier

    def forward(self, x):
        x = x * self.multiplier
        return (x - self.mean) / self.std
