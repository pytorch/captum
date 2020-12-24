from typing import Optional

import numpy as np


class BlendAlpha(object):
    """
    NumPy version of the BlendAlpha transform.

    Args:
        background (array, optional):  An NCHW image array to be used as the
            Alpha channel's background.
    """

    def __init__(self, background: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.background = background

    def blend_alpha(self, x: np.ndarray) -> np.ndarray:
        """
        Blend the Alpha channel into the RGB channels.
        Arguments:
            x (array): RGBA image array to blend into an RGB image array.
        Returns:
            blended (array): RGB image array.
        """
        assert x.shape[1] == 4
        assert x.ndim == 4
        rgb, alpha = x[:, :3, ...], x[:, 3:4, ...]
        background = (
            self.background
            if self.background is not None
            else np.random.randn(*rgb.shape)
        )
        blended = alpha * rgb + (1 - alpha) * background
        return blended


class RandomSpatialJitter(object):
    """
    NumPy version of the RandomSpatialJitter transform.

    Arguments:
        translate (int):  The amount to translate the H and W dimensions
            of an CHW or NCHW array.
    """

    def __init__(self, translate: int) -> None:
        super().__init__()
        self.pad_range = translate

    def translate_array(self, x: np.ndarray, insets) -> np.ndarray:
        x = np.pad(x, (self.pad_range, self.pad_range), "reflect")
        x = np.roll(x, (self.pad_range - insets[1]), axis=0)
        x = np.roll(x, (self.pad_range - insets[0]), axis=1)

        h_crop = x.shape[0] - (self.pad_range * 2)
        w_crop = x.shape[1] - (self.pad_range * 2)
        sw, sh = x.shape[1] // 2 - (w_crop // 2), x.shape[0] // 2 - (h_crop // 2)
        x = x[..., sh : sh + h_crop, sw : sw + w_crop]
        return x

    def jitter(self, x: np.ndarray) -> np.ndarray:
        insets = (
            np.random.randint(high=self.pad_range),
            np.random.randint(high=self.pad_range),
        )
        return self.translate_array(x, insets)


class CenterCrop(object):
    """
    NumPy version of the CenterCrop transform.

    Arguments:
        size (int, sequence) or (int): Number of pixels to center crop away.
    """

    def __init__(self, size=0) -> None:
        super().__init__()
        if type(size) is list or type(size) is tuple:
            assert len(size) == 2, (
                "CenterCrop requires a single crop value or a tuple of (height,width)"
                + "in pixels for cropping."
            )
            self.crop_val = size
        else:
            self.crop_val = [size] * 2
        assert len(self.crop_val) == 2

    def crop(self, input: np.ndarray) -> np.ndarray:
        """
        Center crop an input.
        Arguments:
            input (array): Input to center crop.
        Returns:
            cropped input (array): A center cropped array.
        """

        assert input.ndim == 3 or input.ndim == 4
        if input.ndim == 4:
            h, w = input.shape[2], input.shape[3]
        elif input.ndim == 3:
            h, w = input.shape[1], input.shape[2]
        h_crop = h - self.crop_val[0]
        w_crop = w - self.crop_val[1]
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        return input[..., sh : sh + h_crop, sw : sw + w_crop]


class ToRGB(object):
    """
    NumPy version of the ToRGB transform
    """

    @staticmethod
    def klt_transform() -> np.ndarray:
        KLT = [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        transform = np.array(KLT, dtype=float)
        transform = transform / np.linalg.norm(transform, axis=0).max()
        return transform

    @staticmethod
    def i1i2i3_transform() -> np.ndarray:
        i1i2i3_matrix = [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 2, 0, -1 / 2],
            [-1 / 4, 1 / 2, -1 / 4],
        ]
        return np.array(i1i2i3_matrix, dtype=float)

    def __init__(self, transform_name: str = "klt") -> None:
        super().__init__()

        if transform_name == "klt":
            self.transform = ToRGB.klt_transform()
        elif transform_name == "i1i2i3":
            self.transform = ToRGB.i1i2i3_transform()
        else:
            raise ValueError("transform_name has to be either 'klt' or 'i1i2i3'")

    def to_rgb(self, x: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Args:
            x (array):  A CHW or NCHW RGB or RGBA image array.
            inverse (bool):  Whether to recorrelate or decorrelate colors.
        Returns:
            *array*:  An array with it's colors recorrelated or decorrelated.
        """

        assert x.ndim == 3 or x.ndim == 4

        # alpha channel is taken off...
        if x.ndim == 3:
            has_alpha = x.shape[0] == 4
            h, w = x.shape[1], x.shape[2]
        elif x.ndim == 4:
            has_alpha = x.shape[1] == 4
            h, w = x.shape[2], x.shape[3]
        if has_alpha:
            if x.ndim == 3:
                x, alpha_channel = x[:3], x[3:]
            elif x.ndim == 4:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.ndim == alpha_channel.ndim  # ensure we "keep_dim"

        if x.ndim == 3:
            flat = x.reshape(x.shape[1], h * w)
        elif x.ndim == 4:
            flat = x.reshape(x.shape[0], x.shape[1], h * w)

        if inverse:
            correct = np.linalg.inv(self.transform) @ flat
        else:
            correct = self.transform @ flat

        if x.ndim == 3:
            chw = correct.reshape(x.shape[1], h, w)
        elif x.ndim == 4:
            chw = correct.reshape(x.shape[0], x.shape[1], h, w)

        # ...alpha channel is concatenated on again.
        if has_alpha:
            d = 0 if x.ndim == 3 else 1
            chw = np.concatenate([chw, alpha_channel], d)

        return chw
