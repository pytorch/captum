#!/usr/bin/env python3
import unittest

import torch

from captum.optim._param.image import images
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestFFTImage(BaseTest):
    def test_pytorch_fftfreq(self) -> None:
        assertTensorAlmostEqual(
            self,
            images.FFTImage.pytorch_fftfreq(4, 4),
            torch.tensor([0.0000, 0.0625, -0.1250, -0.0625]),
            0,
        )

    def test_rfft2d_freqs(self) -> None:
        assertTensorAlmostEqual(
            self,
            images.FFTImage.rfft2d_freqs(height=2, width=3),
            torch.tensor([[0.0000, 0.3333, 0.3333], [0.5000, 0.6009, 0.6009]]),
            delta=0.0002,
        )


class TestPixelImage(BaseTest):
    def test_pixelimage_random(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage random due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        image_param = images.PixelImage(size=size, channels=channels)

        self.assertEqual(image_param.image.dim(), 4)
        self.assertEqual(image_param.image.size(0), 1)
        self.assertEqual(image_param.image.size(1), channels)
        self.assertEqual(image_param.image.size(2), size[0])
        self.assertEqual(image_param.image.size(3), size[1])

    def test_pixelimage_init(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage init due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        init_tensor = torch.randn(3, 224, 224)
        image_param = images.PixelImage(size=size, channels=channels, init=init_tensor)

        self.assertEqual(image_param.image.dim(), 4)
        self.assertEqual(image_param.image.size(0), 1)
        self.assertEqual(image_param.image.size(1), channels)
        self.assertEqual(image_param.image.size(2), size[0])
        self.assertEqual(image_param.image.size(3), size[1])
        assertTensorAlmostEqual(self, image_param.image, init_tensor, 0)


if __name__ == "__main__":
    unittest.main()
