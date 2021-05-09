from copy import deepcopy
from types import MethodType
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image
except (ImportError, AssertionError):
    print("The Pillow/PIL library is required to use Captum's Optim library")

from captum.optim._param.image.transforms import SymmetricPadding, ToRGB
from captum.optim._utils.image.common import save_tensor_as_image, show

TORCH_VERSION = torch.__version__


class ImageTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls: Type["ImageTensor"],
        x: Union[List, np.ndarray, torch.Tensor] = [],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor) and x.is_cuda:
            x.show = MethodType(cls.show, x)
            x.export = MethodType(cls.export, x)
            return x
        else:
            return super().__new__(cls, x, *args, **kwargs)

    @classmethod
    def open(cls, path: str, scale: float = 255.0) -> "ImageTensor":
        if path.startswith("https://") or path.startswith("http://"):
            response = requests.get(path, stream=True)
            img = Image.open(response.raw)
        else:
            img = Image.open(path)
        img_np = np.array(img.convert("RGB")).astype(np.float32)
        return cls(img_np.transpose(2, 0, 1) / scale)

    def __repr__(self) -> str:
        prefix = "ImageTensor("
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self, indent)
        suffixes = []
        if self.device.type != torch._C._get_default_device() or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        ):
            suffixes.append("device='" + str(self.device) + "'")
        return torch._tensor_str._add_suffixes(
            prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse
        )

    @classmethod
    def __torch_function__(
        cls: Type["ImageTensor"],
        func: Callable,
        types: List[Type[torch.Tensor]],
        args: Tuple = (),
        kwargs: dict = None,
    ) -> torch.Tensor:
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def show(
        self, figsize: Optional[Tuple[int, int]] = None, scale: float = 255.0
    ) -> None:
        show(self, figsize=figsize, scale=scale)

    def export(self, filename: str, scale: float = 255.0) -> None:
        save_tensor_as_image(self, filename=filename, scale=scale)


class InputParameterization(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        raise NotImplementedError


class ImageParameterization(InputParameterization):
    pass


class FFTImage(ImageParameterization):
    """
    Parameterize an image using inverse real 2D FFT

    Args:
        size (list of int): The H & W dimensions to use for creating
            the nn.Parameter tensor.
        channels (int): The number of channels to create.
        batch (int): The number of batches to create.
        init (torch.tensor, optional): Optionally specify a tensor to
            use instead of creating one.
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if init is None:
            assert len(size) == 2
            self.size = size
        else:
            assert init.dim() == 3 or init.dim() == 4
            if init.dim() == 3:
                init = init.unsqueeze(0)
            self.size = (init.size(2), init.size(3))
        self.torch_rfft, self.torch_irfft, self.torch_fftfreq = self.get_fft_funcs()

        frequencies = self.rfft2d_freqs(*self.size)
        scale = 1.0 / torch.max(
            frequencies,
            torch.full_like(frequencies, 1.0 / (max(self.size[0], self.size[1]))),
        )
        scale = scale * ((self.size[0] * self.size[1]) ** (1 / 2))
        spectrum_scale = scale[None, :, :, None]
        self.register_buffer("spectrum_scale", spectrum_scale)

        if init is None:
            coeffs_shape = (
                batch,
                channels,
                self.size[0],
                self.size[1] // 2 + 1,
                2,
            )
            random_coeffs = torch.randn(
                coeffs_shape
            )  # names=["C", "H_f", "W_f", "complex"]
            fourier_coeffs = random_coeffs / 50
        else:
            fourier_coeffs = self.torch_rfft(init) / spectrum_scale

        self.fourier_coeffs = nn.Parameter(fourier_coeffs)

    def rfft2d_freqs(self, height: int, width: int) -> torch.Tensor:
        """
        Computes 2D spectrum frequencies.

        Args:
            height (int): The h dimension of the 2d frequency scale.
            width (int): The w dimension of the 2d frequency scale.
        Returns:
            tensor (tensor): A 2d frequency scale tensor.
        """

        fy = self.torch_fftfreq(height)[:, None]
        # on odd input dimensions we need to keep one additional frequency
        wadd = 2 if width % 2 == 1 else 1
        fx = self.torch_fftfreq(width)[: width // 2 + wadd]
        return torch.sqrt((fx * fx) + (fy * fy))

    def get_fft_funcs(self) -> Tuple[Callable, Callable, Callable]:
        """
        Support older versions of PyTorch.

        Returns:
            fft functions (tuple of Callable): A list of FFT functions
                to use for irfft and rfft operations.
        """

        if TORCH_VERSION >= "1.7.0":
            import torch.fft

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.view_as_real(torch.fft.rfftn(x, s=self.size))

            def torch_irfft(x: torch.Tensor) -> torch.Tensor:
                if type(x) is not torch.complex64:
                    x = torch.view_as_complex(x)
                return torch.fft.irfftn(x, s=self.size)  # type: ignore

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                return torch.fft.fftfreq(v, d)

        else:
            import torch

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.rfft(x, signal_ndim=2)

            def torch_irfft(x: torch.Tensor) -> torch.Tensor:
                return torch.irfft(x, signal_ndim=2)[
                    :, :, : self.size[0], : self.size[1]
                ]

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                """PyTorch version of np.fft.fftfreq"""
                results = torch.empty(v)
                s = (v - 1) // 2 + 1
                results[:s] = torch.arange(0, s)
                results[s:] = torch.arange(-(v // 2), 0)
                return results * (1.0 / (v * d))

        return torch_rfft, torch_irfft, torch_fftfreq

    def forward(self) -> torch.Tensor:
        """
        Returns:
            spatially correlated tensor (tensor): A spatially recorrelated tensor.
        """

        h, w = self.size
        scaled_spectrum = self.fourier_coeffs * self.spectrum_scale
        output = self.torch_irfft(scaled_spectrum)
        return output.refine_names("B", "C", "H", "W")


class PixelImage(ImageParameterization):
    """
    Parameterize a simple image tensor.

    Args:
        size (list of int): The H & W dimensions to use for creating
            the nn.Parameter tensor.
        channels (int): The number of channels to create.
        batch (int): The number of batches to create.
        init (torch.tensor, optional): Optionally specify a tensor to
            use instead of creating one.
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if init is None:
            assert size is not None and channels is not None and batch is not None
            init = torch.randn([batch, channels, size[0], size[1]]) / 10 + 0.5
        else:
            assert init.dim() == 3 or init.dim() == 4
            if init.dim() == 3:
                init = init.unsqueeze(0)
            assert init.shape[1] == 3, "PixelImage init should have 3 channels, "
            f"input has {init.shape[1]} channels."
        self.image = nn.Parameter(init)

    def forward(self) -> torch.Tensor:
        return self.image.refine_names("B", "C", "H", "W")


class LaplacianImage(ImageParameterization):
    """
    Parameterize an image with a laplacian pyramid.

    Args:
        size (list of int): The H & W dimensions to use for creating
            the nn.Parameter tensor.
        channels (int): The number of channels to create.
        batch (int): The number of batches to create.
        init (torch.tensor, optional): Optionally specify a tensor to
            use instead of creating one.
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        power = 0.1

        if init is None:
            tensor_params, self.scaler = self.setup_input(size, channels, power, init)

            self.tensor_params = torch.nn.ModuleList(
                [deepcopy(tensor_params) for b in range(batch)]
            )
        else:
            init = init.unsqueeze(0) if init.dim() == 3 else init
            P = []
            for b in range(init.size(0)):
                tensor_params, self.scaler = self.setup_input(
                    size, channels, power, init[b].unsqueeze(0)
                )
                P.append(tensor_params)
            self.tensor_params = torch.nn.ModuleList(P)

    def setup_input(
        self,
        size: Tuple[int, int],
        channels: int,
        power: float = 0.1,
        init: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.nn.Upsample]]:
        tensor_params, scaler = [], []
        scale_list = [1, 2, 4, 8, 16, 32]
        for scale in scale_list:
            h, w = int(size[0] // scale), int(size[1] // scale)
            if init is None:
                x = torch.randn([1, channels, h, w]) / 10
            else:
                x = F.interpolate(init.clone(), size=(h, w), mode="bilinear")
                x = x / 6  # Prevents output from being all white
            upsample = torch.nn.Upsample(scale_factor=scale, mode="nearest")
            x = x * (scale ** power) / (32 ** power)
            x = torch.nn.Parameter(x)
            tensor_params.append(x)
            scaler.append(upsample)
        tensor_params = torch.nn.ParameterList(tensor_params)
        return tensor_params, scaler

    def create_tensor(self, params_list: torch.nn.ParameterList) -> torch.Tensor:
        A = []
        for xi, upsamplei in zip(params_list, self.scaler):
            A.append(upsamplei(xi))
        return torch.sum(torch.cat(A), 0) + 0.5

    def forward(self) -> torch.Tensor:
        A = []
        for params_list in self.tensor_params:
            tensor = self.create_tensor(params_list)
            A.append(tensor)
        return torch.stack(A).refine_names("B", "C", "H", "W")


class SharedImage(ImageParameterization):
    """
    Share some image parameters across the batch to increase spatial alignment,
    by using interpolated lower resolution tensors.
    This is sort of like a laplacian pyramid but more general.

    Offsets are similar to phase in Fourier transforms, and can be applied to
    any dimension.

    Mordvintsev, et al., "Differentiable Image Parameterizations", Distill, 2018.
    https://distill.pub/2018/differentiable-parameterizations/

    Args:
        shapes (list of int or list of list of ints): The shapes of the shared tensors
            to use for creating the nn.Parameter tensors.
        parameterization (ImageParameterization):  An image parameterization instance.
        offset (int or list of int or list of list of ints): The offsets to use for the
            shared tensors.
    """

    def __init__(
        self,
        shapes: Union[Tuple[Tuple[int]], Tuple[int]] = None,
        parameterization: ImageParameterization = None,
        offset: Union[int, Tuple[int], Tuple[Tuple[int]], None] = None,
    ) -> None:
        super().__init__()
        assert shapes is not None
        A = []
        shared_shapes = [shapes] if type(shapes[0]) is not tuple else shapes
        for shape in shared_shapes:
            assert len(shape) >= 2 and len(shape) <= 4
            shape = ([1] * (4 - len(shape))) + list(shape)
            batch, channels, height, width = shape
            A.append(torch.nn.Parameter(torch.randn([batch, channels, height, width])))
        self.shared_init = torch.nn.ParameterList(A)
        self.parameterization = parameterization
        self.offset = self.get_offset(offset, len(A)) if offset is not None else None

    def get_offset(self, offset: Union[int, Tuple[int]], n: int) -> List[List[int]]:
        if type(offset) is tuple or type(offset) is list:
            if type(offset[0]) is tuple or type(offset[0]) is list:
                assert len(offset) == n and all(len(t) == 4 for t in offset)
            else:
                assert len(offset) >= 1 and len(offset) <= 4
                offset = [([0] * (4 - len(offset))) + list(offset)] * n
        else:
            offset = [[offset] * 4] * n
        offset = [list(v) for v in offset]
        assert all([all([type(o) is int for o in v]) for v in offset])
        return offset

    def apply_offset(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply list of offsets to list of tensors.

        Args:
            x_list (list of torch.Tensor): list of tensors to offset.
        Returns:
            A (list of torch.Tensor): list of offset tensors.
        """

        A = []
        for x, offset in zip(x_list, self.offset):
            assert x.dim() == 4
            size = list(x.size())

            offset_pad = (
                [[abs(offset[0])] * 2]
                + [[abs(offset[1])] * 2]
                + [[abs(offset[2])] * 2]
                + [[abs(offset[3])] * 2]
            )

            x = SymmetricPadding.apply(x, offset_pad)

            for o, s in zip(offset, range(x.dim())):
                x = torch.roll(x, shifts=o, dims=s)

            x = x[: size[0], : size[1], : size[2], : size[3]]
            A.append(x)
        return A

    def interpolate_tensor(
        self, x: torch.Tensor, batch: int, channels: int, height: int, width: int
    ) -> torch.Tensor:
        """
        Linear interpolation for 4D, 5D, and 6D tensors.
        If the batch dimension needs to be resized,
        we move it's location temporarily for F.interpolate.
        """

        if x.size(1) == channels:
            mode = "bilinear"
            size = (height, width)
        else:
            mode = "trilinear"
            x = x.unsqueeze(0)
            size = (channels, height, width)
        x = F.interpolate(x, size=size, mode=mode)
        x = x.squeeze(0) if len(size) == 3 else x
        if x.size(0) != batch:
            x = x.permute(1, 0, 2, 3)
            x = F.interpolate(
                x.unsqueeze(0),
                size=(batch, x.size(2), x.size(3)),
                mode="trilinear",
            ).squeeze(0)
            x = x.permute(1, 0, 2, 3)
        return x

    def forward(self) -> torch.Tensor:
        image = self.parameterization()
        x = [
            self.interpolate_tensor(
                shared_tensor,
                image.size(0),
                image.size(1),
                image.size(2),
                image.size(3),
            )
            for shared_tensor in self.shared_init
        ]
        if self.offset is not None:
            x = self.apply_offset(x)
        return (image + sum(x)).refine_names("B", "C", "H", "W")


class NaturalImage(ImageParameterization):
    r"""Outputs an optimizable input image.

    By convention, single images are CHW and float32s in [0,1].
    The underlying parameterization can be decorrelated via a ToRGB transform.
    When used with the (default) FFT parameterization, this results in a fully
    uncorrelated image parameterization. :-)

    If a model requires a normalization step, such as normalizing imagenet RGB values,
    or rescaling to [0,255], it can perform those steps with the provided transforms or
    inside its computation.

    Arguments:
        size (Tuple[int, int]): The height and width to use for the nn.Parameter image
            tensor.
        channels (int): The number of channels to use when creating the
            nn.Parameter tensor.
        batch (int): The number of channels to use when creating the
            nn.Parameter tensor, or stacking init images.
        parameterization (ImageParameterization, optional): An image parameterization
            class.
        squash_func (Callable[[torch.Tensor], torch.Tensor]], optional): The squash
            function to use after color recorrelation. A funtion or lambda function.
        decorrelation_module (nn.Module, optional): A ToRGB instance.
        decorrelate_init (bool, optional): Whether or not to apply color decorrelation
            to the init tensor input.
    """

    def __init__(
        self,
        size: Tuple[int, int] = [224, 224],
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
        parameterization: ImageParameterization = FFTImage,
        squash_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decorrelation_module: Optional[nn.Module] = ToRGB(transform="klt"),
        decorrelate_init: bool = True,
    ) -> None:
        super().__init__()
        self.decorrelate = decorrelation_module
        if init is not None:
            assert init.dim() == 3 or init.dim() == 4
            if decorrelate_init:
                assert self.decorrelate is not None
                init = (
                    init.refine_names("B", "C", "H", "W")
                    if init.dim() == 4
                    else init.refine_names("C", "H", "W")
                )
                init = self.decorrelate(init, inverse=True).rename(None)
            if squash_func is None:

                def squash_func(x: torch.Tensor) -> torch.Tensor:
                    return x.clamp(0, 1)

        else:
            if squash_func is None:

                squash_func = torch.sigmoid

        self.squash_func = squash_func
        self.parameterization = parameterization(
            size=size, channels=channels, batch=batch, init=init
        )

    def forward(self) -> torch.Tensor:
        image = self.parameterization()
        if self.decorrelate is not None:
            image = self.decorrelate(image)
        image = image.rename(None)  # TODO: the world is not yet ready
        return ImageTensor(self.squash_func(image))


__all__ = [
    "ImageTensor",
    "InputParameterization",
    "ImageParameterization",
    "FFTImage",
    "PixelImage",
    "LaplacianImage",
    "SharedImage",
    "NaturalImage",
]
