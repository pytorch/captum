from typing import List

import numpy as np


def weights_to_heatmap_2d(
    array: np.ndarray,
    colors: List[str] = ["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"],
) -> np.ndarray:
    """
    Create a color heatmap of an input weight array.
    By default red represents excitatory values,
    blue represents inhibitory values, and white represents
    no excitation or inhibition.
    """

    assert array.ndim == 2
    assert len(colors) == 5

    def get_color(x: str) -> np.ndarray:
        def hex2base10(x: str) -> float:
            return int(x, 16) / 255.0

        return np.array([hex2base10(x[0:2]), hex2base10(x[2:4]), hex2base10(x[4:6])])

    t_colors = [get_color(c) for c in colors]
    xt = array[None, :].repeat(3, 0).transpose(1, 2, 0)

    color_array = (
        (xt >= 0) * (xt < 0.5) * ((1 - xt * 2) * t_colors[2] + xt * 2 * t_colors[3])
        + (xt >= 0)
        * (xt >= 0.5)
        * ((1 - (xt - 0.5) * 2) * t_colors[3] + (xt - 0.5) * 2 * t_colors[4])
        + (xt < 0)
        * (xt > -0.5)
        * ((1 - (-xt * 2)) * t_colors[2] + (-xt * 2) * t_colors[1])
        + (xt < 0)
        * (xt <= -0.5)
        * ((1 - (-xt - 0.5) * 2) * t_colors[1] + (-xt - 0.5) * 2 * t_colors[0])
    ).transpose(2, 0, 1)
    return color_array
