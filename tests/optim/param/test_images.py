#!/usr/bin/env python3
import unittest

import numpy as np
import torch

from captum.optim._param.image import images
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.optim.helpers import numpy_image


class TestSetupBatch(BaseTest):
    def test_setup_batch_chw(self) -> None:
        init = torch.randn(3, 4, 4)

        batch_test = images.ImageParameterization()
        tensor_wbatch = batch_test.setup_batch(init)
        array_wbatch = numpy_image.setup_batch(init.numpy())

        assertArraysAlmostEqual(tensor_wbatch.numpy(), array_wbatch)

    def test_setup_batch_chwr(self) -> None:
        init = torch.randn(3, 4, 4, 2)

        batch_test = images.ImageParameterization()
        tensor_wbatch = batch_test.setup_batch(init, dim=4)
        array_wbatch = numpy_image.setup_batch(init.numpy(), dim=4)

        assertArraysAlmostEqual(tensor_wbatch.numpy(), array_wbatch)

    def test_setup_batch_init(self) -> None:
        init = torch.randn(5, 3, 4, 4)

        batch_test = images.ImageParameterization()
        tensor_wbatch = batch_test.setup_batch(init, dim=3)
        array_wbatch = numpy_image.setup_batch(init.numpy(), dim=3)

        assertArraysAlmostEqual(tensor_wbatch.numpy(), array_wbatch)


class TestFFTImage(BaseTest):
    def test_pytorch_fftfreq(self) -> None:
        assertArraysAlmostEqual(
            images.FFTImage.pytorch_fftfreq(4, 4).numpy(), np.fft.fftfreq(4, 4)
        )

    def test_rfft2d_freqs(self) -> None:
        height = 2
        width = 3
        assertArraysAlmostEqual(
            images.FFTImage.rfft2d_freqs(height, width).numpy(),
            numpy_image.FFTImage.rfft2d_freqs(height, width),
        )

    def test_fftimage_forward_randn_init(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)

        fftimage = images.FFTImage(size=size)
        fftimage_np = numpy_image.FFTImage(size=size)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_randn_batch(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        batch = 5

        fftimage = images.FFTImage(size=size, batch=batch)
        fftimage_np = numpy_image.FFTImage(size=size, batch=batch)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_randn_channels(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 4

        fftimage = images.FFTImage(size=size, channels=channels)
        fftimage_np = numpy_image.FFTImage(size=size, channels=channels)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_chw(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        init_tensor = torch.randn(3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_bchw(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        init_tensor = torch.randn(1, 3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_batch(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        batch = 5
        init_tensor = torch.randn(1, 3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, batch=batch, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, batch=batch, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)


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

    def test_pixelimage_random_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage random due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        image_param = images.PixelImage(size=size, channels=channels)
        test_tensor = image_param.forward().rename(None)

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), 1)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_pixelimage_init_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage init due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        init_tensor = torch.randn(3, 224, 224)
        image_param = images.PixelImage(size=size, channels=channels, init=init_tensor)
        test_tensor = image_param.forward().rename(None)

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), 1)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])
        assertTensorAlmostEqual(self, test_tensor, init_tensor.squeeze(0), 0)


class TestLaplacianImage(BaseTest):
    def test_laplacianimage_random_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage random due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        image_param = images.LaplacianImage(size=size, channels=channels)
        test_tensor = image_param.forward()

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), 1)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])


if __name__ == "__main__":
    unittest.main()
