import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import signal


def signal_to_spectrogram(data: np.array, rate: int, size: int = 32) -> BytesIO:
    if len(data.shape) != 1:
        raise ValueError(f"audio data must be a 1D array, got shape {data.shape}")
    if data.dtype != np.float32:
        raise ValueError(f"audio data must be a float32 array, got {data.dtype}")

    matplotlib.use("Agg")
    width, height = size, size
    n = data.shape[0]
    t = np.arange(n) / float(rate)
    f, t, Sxx = signal.spectrogram(data, rate)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.pcolormesh(t, f, Sxx, shading="gouraud")
    ax.set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    img_io = BytesIO()
    fig.savefig(img_io, dpi=1)
    img_io.seek(0)
    return img_io
