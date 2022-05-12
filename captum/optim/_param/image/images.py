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
        """
        Args:

            x (list or np.ndarray or torch.Tensor): A list, NumPy array, or PyTorch
                tensor to create an `ImageTensor` from.

        Returns:
           x (ImageTensor): An `ImageTensor` instance.
        """
        if isinstance(x, torch.Tensor) and x.is_cuda:
            x.show = MethodType(cls.show, x)
            x.export = MethodType(cls.export, x)
            return x
        else:
            return super().__new__(cls, x, *args, **kwargs)

    @classmethod
    def open(cls, path: str, scale: float = 255.0, mode: str = "RGB") -> "ImageTensor":
        """
        Load an image file from a URL or local filepath directly into an `ImageTensor`.

        Args:

            path (str): A URL or filepath to an image.
            scale (float, optional): The image scale to use.
                Default: 255.0
            mode (str, optional): The image loading mode to use.
                Default: "RGB"

        Returns:
           x (ImageTensor): An `ImageTensor` instance.
        """
        if path.startswith("https://") or path.startswith("http://"):
            response = requests.get(path, stream=True)
            img = Image.open(response.raw)
        else:
            img = Image.open(path)
        img_np = np.array(img.convert(mode)).astype(np.float32)
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
        """
        Display an `ImageTensor`.

        Args:

            figsize (Tuple[int, int], optional): height & width to use
                for displaying the `ImageTensor` figure.
            scale (float, optional): Value to multiply the `ImageTensor` by so that
                it's value range is [0-255] for display.
                Default: 255.0
        """
        show(self, figsize=figsize, scale=scale)

    def export(self, filename: str, scale: float = 255.0) -> None:
        """
        Save an `ImageTensor` as an image file.

        Args:

            filename (str): The filename to use when saving the `ImageTensor` as an
                image file.
            scale (float, optional): Value to multiply the `ImageTensor` by so that
                it's value range is [0-255] for saving.
                Default: 255.0
        """
        save_tensor_as_image(self, filename=filename, scale=scale)


class InputParameterization(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        raise NotImplementedError


class ImageParameterization(InputParameterization):
    pass


class FFTImage(ImageParameterization):
    """
    Parameterize an image using inverse real 2D FFT
    """

    __constants__ = ["size", "_supports_is_scripting"]

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:

            size (Tuple[int, int]): The height & width dimensions to use for the
                parameterized output image tensor.
            channels (int, optional): The number of channels to use for each image.
                Default: 3
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: 1
            init (torch.tensor, optional): Optionally specify a tensor to
                use instead of creating one.
                Default: None
        """
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
            spectrum_scale = spectrum_scale.to(init.device)
            fourier_coeffs = self.torch_rfft(init) / spectrum_scale

        self.register_buffer("spectrum_scale", spectrum_scale)
        self.fourier_coeffs = nn.Parameter(fourier_coeffs)

        # Check & store whether or not we can use torch.jit.is_scripting()
        self._supports_is_scripting = torch.__version__ >= "1.6.0"

    def rfft2d_freqs(self, height: int, width: int) -> torch.Tensor:
        """
        Computes 2D spectrum frequencies.

        Args:

            height (int): The h dimension of the 2d frequency scale.
            width (int): The w dimension of the 2d frequency scale.

        Returns:
            **tensor** (tensor): A 2d frequency scale tensor.
        """

        fy = self.torch_fftfreq(height)[:, None]
        fx = self.torch_fftfreq(width)[: width // 2 + 1]
        return torch.sqrt((fx * fx) + (fy * fy))

    @torch.jit.export
    def torch_irfftn(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.complex64:
            x = torch.view_as_complex(x)
        return torch.fft.irfftn(x, s=self.size)  # type: ignore

    def get_fft_funcs(self) -> Tuple[Callable, Callable, Callable]:
        """
        Support older versions of PyTorch. This function ensures that the same FFT
        operations are carried regardless of whether your PyTorch version has the
        torch.fft update.

        Returns:
            fft functions (tuple of Callable): A list of FFT functions
                to use for irfft, rfft, and fftfreq operations.
        """

        if TORCH_VERSION > "1.7.0":
            if TORCH_VERSION <= "1.8.0":
                global torch
                import torch.fft

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.view_as_real(torch.fft.rfftn(x, s=self.size))

            torch_irfftn = self.torch_irfftn

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                return torch.fft.fftfreq(v, d)

        else:

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.rfft(x, signal_ndim=2)

            def torch_irfftn(x: torch.Tensor) -> torch.Tensor:
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

        return torch_rfft, torch_irfftn, torch_fftfreq

    def forward(self) -> torch.Tensor:
        """
        Returns:
            **output** (torch.tensor): A spatially recorrelated tensor.
        """

        scaled_spectrum = self.fourier_coeffs * self.spectrum_scale
        output = self.torch_irfft(scaled_spectrum)
        if self._supports_is_scripting:
            if torch.jit.is_scripting():
                return output
        return output.refine_names("B", "C", "H", "W")


class PixelImage(ImageParameterization):
    """
    Parameterize a simple pixel image tensor that requires no additional transforms.
    """

    __constants__ = ["_supports_is_scripting"]

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:

            size (Tuple[int, int]): The height & width dimensions to use for the
                parameterized output image tensor.
            channels (int, optional): The number of channels to use for each image.
                Default: 3
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: 1
            init (torch.tensor, optional): Optionally specify a tensor to
                use instead of creating one.
                Default: None
        """
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

        # Check & store whether or not we can use torch.jit.is_scripting()
        self._supports_is_scripting = torch.__version__ >= "1.6.0"

    def forward(self) -> torch.Tensor:
        if self._supports_is_scripting:
            if torch.jit.is_scripting():
                return self.image
        return self.image.refine_names("B", "C", "H", "W")


class LaplacianImage(ImageParameterization):
    """
    TODO: Fix divison by 6 in setup_input when init is not None.
    Parameterize an image tensor with a laplacian pyramid.
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:

            size (Tuple[int, int]): The height & width dimensions to use for the
                parameterized output image tensor.
            channels (int, optional): The number of channels to use for each image.
                Default: 3
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: 1
            init (torch.tensor, optional): Optionally specify a tensor to
                use instead of creating one.
                Default: None
        """
        super().__init__()
        power = 0.1

        if init is None:
            tensor_params, self.scaler = self._setup_input(size, channels, power, init)

            self.tensor_params = torch.nn.ModuleList(
                [deepcopy(tensor_params) for b in range(batch)]
            )
        else:
            init = init.unsqueeze(0) if init.dim() == 3 else init
            P = []
            for b in range(init.size(0)):
                tensor_params, self.scaler = self._setup_input(
                    size, channels, power, init[b].unsqueeze(0)
                )
                P.append(tensor_params)
            self.tensor_params = torch.nn.ModuleList(P)

    def _setup_input(
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

    def _create_tensor(self, params_list: torch.nn.ParameterList) -> torch.Tensor:
        """
        Resize tensor parameters to the target size.

        Args:

            params_list (torch.nn.ParameterList): List of tensors to resize.

        Returns:
            **tensor** (torch.Tensor): The sum of all tensor parameters.
        """
        A: List[torch.Tensor] = []
        for xi, upsamplei in zip(params_list, self.scaler):
            A.append(upsamplei(xi))
        return torch.sum(torch.cat(A), 0) + 0.5

    def forward(self) -> torch.Tensor:
        A: List[torch.Tensor] = []
        for params_list in self.tensor_params:
            tensor = self._create_tensor(params_list)
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
    """

    def __init__(
        self,
        shapes: Union[Tuple[Tuple[int]], Tuple[int]] = None,
        parameterization: ImageParameterization = None,
        offset: Union[int, Tuple[int], Tuple[Tuple[int]], None] = None,
    ) -> None:
        """
        Args:

            shapes (list of int or list of list of ints): The shapes of the shared
                tensors to use for creating the nn.Parameter tensors.
            parameterization (ImageParameterization): An image parameterization
                instance.
            offset (int or list of int or list of list of ints , optional): The offsets
                to use for the shared tensors.
                Default: None
        """
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
        self.offset = self._get_offset(offset, len(A)) if offset is not None else None

    def _get_offset(self, offset: Union[int, Tuple[int]], n: int) -> List[List[int]]:
        """
        Given offset values, return a list of offsets for _apply_offset to use.

        Args:

            offset (int or list of int or list of list of ints , optional): The offsets
                to use for the shared tensors.
            n (int): The number of tensors needing offset values.

        Returns:
            **offset** (list of list of int): A list of offset values.
        """
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

    def _apply_offset(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply list of offsets to list of tensors.

        Args:

            x_list (list of torch.Tensor): list of tensors to offset.

        Returns:
            **A** (list of torch.Tensor): list of offset tensors.
        """

        A: List[torch.Tensor] = []
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

    def _interpolate_tensor(
        self, x: torch.Tensor, batch: int, channels: int, height: int, width: int
    ) -> torch.Tensor:
        """
        Linear interpolation for 4D, 5D, and 6D tensors. If the batch dimension needs
        to be resized, we move it's location temporarily for F.interpolate.

        Args:

            x (torch.Tensor): The tensor to resize.
            batch (int): The batch size to resize the tensor to.
            channels (int): The channel size to resize the tensor to.
            height (int): The height to resize the tensor to.
            width (int): The width to resize the tensor to.

        Returns:
            **tensor** (torch.Tensor): A resized tensor.
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
            self._interpolate_tensor(
                shared_tensor,
                image.size(0),
                image.size(1),
                image.size(2),
                image.size(3),
            )
            for shared_tensor in self.shared_init
        ]
        if self.offset is not None:
            x = self._apply_offset(x)
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
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
        parameterization: ImageParameterization = FFTImage,
        squash_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decorrelation_module: Optional[nn.Module] = ToRGB(transform="klt"),
        decorrelate_init: bool = True,
    ) -> None:
        """
        Args:

            size (Tuple[int, int], optional): The height and width to use for the
                nn.Parameter image tensor.
                Default: (224, 224)
            channels (int, optional): The number of channels to use when creating the
                nn.Parameter tensor.
                Default: 3
            batch (int, optional): The number of channels to use when creating the
                nn.Parameter tensor, or stacking init images.
                Default: 1
            parameterization (ImageParameterization, optional): An image
                parameterization class, or instance of an image parameterization class.
                Default: FFTImage
            squash_func (Callable[[torch.Tensor], torch.Tensor]], optional): The squash
                function to use after color recorrelation. A funtion or lambda function.
                Default: None
            decorrelation_module (nn.Module, optional): A ToRGB instance.
                Default: ToRGB
            decorrelate_init (bool, optional): Whether or not to apply color
                decorrelation to the init tensor input.
                Default: True
        """
        super().__init__()
        if not isinstance(parameterization, ImageParameterization):
            # Verify uninitialized class is correct type
            assert issubclass(parameterization, ImageParameterization)
        else:
            assert isinstance(parameterization, ImageParameterization)

        self.decorrelate = decorrelation_module
        if init is not None and not isinstance(parameterization, ImageParameterization):
            assert init.dim() == 3 or init.dim() == 4
            if decorrelate_init and self.decorrelate is not None:
                init = (
                    init.refine_names("B", "C", "H", "W")
                    if init.dim() == 4
                    else init.refine_names("C", "H", "W")
                )
                init = self.decorrelate(init, inverse=True).rename(None)

            if squash_func is None:
                squash_func = self._clamp_image

        self.squash_func = torch.sigmoid if squash_func is None else squash_func
        if not isinstance(parameterization, ImageParameterization):
            parameterization = parameterization(
                size=size, channels=channels, batch=batch, init=init
            )
        self.parameterization = parameterization

    @torch.jit.export
    def _clamp_image(self, x: torch.Tensor) -> torch.Tensor:
        """JIT supported squash function."""
        return x.clamp(0, 1)

    @torch.jit.ignore
    def _to_image_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrap ImageTensor in torch.jit.ignore for JIT support.

        Args:

            x (torch.tensor): An input tensor.

        Returns:
            x (ImageTensor): An instance of ImageTensor with the input tensor.
        """
        return ImageTensor(x)

    def forward(self) -> torch.Tensor:
        image = self.parameterization()
        if self.decorrelate is not None:
            image = self.decorrelate(image)
        image = image.rename(None)  # TODO: the world is not yet ready
        return self._to_image_tensor(self.squash_func(image))


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
