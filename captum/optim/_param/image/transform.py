import logging
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate, scale, shear, translate


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


# class RandomScale(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.scale = torch.distributions.Uniform(0.95, 1.05)

#     def forward(self, x):
#         by = self.scale.sample().item()
#         return F.interpolate(x, scale_factor=by, mode="bilinear")


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
    TODO: Can we look into Distributions more to give more control and
    be more PyTorch-y?
    """

    def __init__(self, rotate=False, scale=False, shear=False, translate=False):
        super().__init__()
        self.rotate = rotate
        self.scale = scale
        self.shear = shear
        self.translate = translate

    def forward(self, x):
        if self.rotate:
            rotate_angle = torch.randn(1, device=x.device)  # >95% < 6deg
            logging.info(f"Rotate: {rotate_angle}")
            x = rotate(x, rotate_angle)
        if self.scale:
            scale_factor = (torch.randn(1, device=x.device) / 40.0) + 1
            logging.info(f"Scale: {scale_factor}")
            x = scale(x, scale_factor)
        if self.shear:
            shear_matrix = torch.randn((1, 2), device=x.device) / 40.0  # >95% < 2deg
            logging.info(f"Shear: {shear_matrix}")
            x = shear(x, shear_matrix)
        if self.translate:
            translation = torch.randn((1, 2), device=x.device)
            logging.info(f"Translate: {translation}")
            x = translate(x, translation)
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
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.as_tensor(mean).view(3, 1, 1).to(device)
        self.std = torch.as_tensor(std).view(3, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std


def test_transform():
    from clarity.pytorch.fixtures import image
    from clarity.pytorch.io import show

    input_image = image()[None, ...]
    show(input_image)
    transform = GaussianSmoothing(channels=3, kernel_size=(5, 5), sigma=2)
    transformed_image = transform(input_image)
    show(transformed_image)
