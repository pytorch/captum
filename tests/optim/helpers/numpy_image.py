from typing import Optional

import numpy as np


def setup_batch(x: np.ndarray, batch: int = 1, dim: int = 3) -> np.ndarray:
    assert batch > 0
    x = x[None, :] if x.ndim == dim and batch == 1 else x
    x = (
        np.stack([np.copy(x) for b in range(batch)])
        if x.ndim == dim and batch > 1
        else x
    )
    return x


class FFTImage:
    """Parameterize an image using inverse real 2D FFT"""

    def __init__(
        self,
        size=None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if init is None:
            assert len(size) == 2
            self.size = size
        else:
            assert init.ndim == 3 or init.ndim == 4
            self.size = (
                (init.shape[1], init.shape[2])
                if init.ndim == 3
                else (init.shape[2], init.shape[3])
            )

        frequencies = FFTImage.rfft2d_freqs(*self.size)
        scale = 1.0 / np.maximum(
            frequencies,
            np.full_like(frequencies, 1.0 / (max(self.size[0], self.size[1]))),
        )
        scale = scale * ((self.size[0] * self.size[1]) ** (1 / 2))
        spectrum_scale = scale[None, :, :, None]
        self.spectrum_scale = spectrum_scale

        if init is None:
            coeffs_shape = (batch, channels, self.size[0], self.size[1] // 2 + 1, 2)
            random_coeffs = np.random.randn(
                *coeffs_shape
            )  # names=["C", "H_f", "W_f", "complex"]
            fourier_coeffs = random_coeffs / 50
        else:
            fourier_coeffs = (
                np.fft.rfftn(init, s=self.size).view("(2,)float") / spectrum_scale
            )

        self.fourier_coeffs = fourier_coeffs

    @staticmethod
    def rfft2d_freqs(height: int, width: int) -> np.ndarray:
        """Computes 2D spectrum frequencies."""
        fy = np.fft.fftfreq(height)[:, None]
        # on odd input dimensions we need to keep one additional frequency
        wadd = 2 if width % 2 == 1 else 1
        fx = np.fft.fftfreq(width)[: width // 2 + wadd]
        return np.sqrt((fx * fx) + (fy * fy))

    def set_image(self, correlated_image: np.ndarray) -> None:
        coeffs = np.fft.rfftn(correlated_image, s=self.size).view("(2,)float")
        self.fourier_coeffs = coeffs / self.spectrum_scale

    def forward(self) -> np.ndarray:
        h, w = self.size
        scaled_spectrum = self.fourier_coeffs * self.spectrum_scale
        scaled_spectrum = scaled_spectrum.astype(complex)
        output = np.fft.irfftn(scaled_spectrum, s=self.size)
        return output.view(dtype=np.complex128)[..., 0].real
