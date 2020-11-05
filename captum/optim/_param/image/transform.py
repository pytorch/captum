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


class CenterCrop(torch.nn.Module):

    def __init__(self, size = 0):
        super(CenterCrop, self).__init__()
        self.crop_val = [size] * 2 if size is not list and size is not tuple else size

    def forward(self, input):
        if input.dim() == 4:
            h, w = input.size(2), input.size(3)         
        elif input.dim() == 3:
            h, w = input.size(1), input.size(2)          
        h_crop = h - self.crop_val[0]
        w_crop = w - self.crop_val[1]
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        return input[:, :, sh:sh + h_crop, sw:sw + w_crop]

    
def rand_select(transform_values):
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
        mode (str): What rescaling method to use.
    """

    def __init__(self, scale, mode='interpolate'):
        super(RandomScale, self).__init__()
        self.scale = scale
        self.mode = mode
       
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
 
    def interpolate_tensor(self, input, scale):
        return torch.nn.functional.interpolate(
            input, scale_factor=scale, mode="bilinear"
        )

    def forward(self, input):
        scale = rand_select(self.scale)
        if self.mode == 'interpolate':
            return self.interpolate_tensor(input, scale=scale)        
        elif self.mode == 'affine_grid':
            return self.scale_tensor(input, scale=scale)

    
class RandomSpatialJitter(torch.nn.Module):

    def __init__(self, jitter_val):
        super(Jitter, self).__init__()
        self.jitter_val = jitter_val

    def forward(self, input):
        h_shift = = torch.randint(low=-self.jitter_val, high=self.jitter_val, size=[1]).item()
        w_shift = torch.randint(low=-self.jitter_val, high=self.jitter_val, size=[1]).item()
        return torch.roll(torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3)


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

    def __init__(self, channels, kernel_size, sigma, dim: int = 2):
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
