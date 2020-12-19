#!/usr/bin/env python3
import unittest

import numpy as np
import torch
import torch.nn.functional as F

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
        batch = 2

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
        assertArraysAlmostEqual(fftimage_tensor.detach().numpy(), fftimage_array, 25.0)

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
        assertArraysAlmostEqual(fftimage_tensor.detach().numpy(), fftimage_array, 25.0)

    def test_fftimage_forward_init_batch(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        batch = 2
        init_tensor = torch.randn(1, 3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, batch=batch, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, batch=batch, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)
        assertArraysAlmostEqual(fftimage_tensor.detach().numpy(), fftimage_array, 25.0)


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
                "Skipping LaplacianImage random due to insufficient Torch version."
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


class TestSharedImage(BaseTest):
    def test_sharedimage_get_offset_single_number(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset = image_param.get_offset(4, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[4, 4, 4, 4]] * 3)

    def test_sharedimage_get_offset_exact(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = ((1, 2, 3, 4), (4, 3, 2, 1), (1, 2, 3, 4))
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[int(o) for o in v] for v in offset_vals])

    def test_sharedimage_apply_offset_single_set_four_numbers(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = (1, 2, 3, 4)
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [list(offset_vals)] * 3)

    def test_sharedimage_apply_offset_single_set_three_numbers(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = (2, 3, 4)
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[0] + list(offset_vals)] * 3)

    def test_sharedimage_apply_offset_single_set_two_numbers(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = (3, 4)
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[0, 0] + list(offset_vals)] * 3)

    def test_apply_offset(self):
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        size = (4, 3, 224, 224)
        shared_shapes = (128 // 2, 128 // 2)
        offset_vals = (2, 3, 4, 5)
        test_param = lambda: torch.ones(*size)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param, offset=offset_vals
        )

        test_x_list = [torch.ones(*size) for x in range(size[0])]
        output_A = image_param.apply_offset(test_x_list, size)

        x_list = [torch.ones(*size) for x in range(size[0])]
        offset_list = image_param.offset
        expected_A = []
        for x, offset in zip(x_list, offset_list):
            x = F.pad(x, offset, "reflect")
            x = x[: size[0], : size[1], : size[2], : size[3]]
            expected_A.append(x)

        for t_expected, t_output in zip(expected_A, output_A):
            assertTensorAlmostEqual(self, t_expected, t_output)

    def test_interpolate_tensor(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        size = (224, 224)
        channels = 3
        batch = 1

        test_tensor = torch.ones(6, 4, 128, 128)
        output_tensor = image_param.interpolate_tensor(
            test_tensor, size, batch, channels
        )

        self.assertEqual(output_tensor.dim(), 4)
        self.assertEqual(output_tensor.size(0), batch)
        self.assertEqual(output_tensor.size(1), channels)
        self.assertEqual(output_tensor.size(2), size[0])
        self.assertEqual(output_tensor.size(3), size[1])

    def test_sharedimage_single_shape_hw_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (128 // 2, 128 // 2)
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertEqual(image_param.shared_init.dim(), 4)
        self.assertEqual(image_param.shared_init.shape, (1, 1, 128 // 2, 128 // 2))
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_single_shape_chw_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (3, 128 // 2, 128 // 2)
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertEqual(image_param.shared_init.dim(), 4)
        self.assertEqual(image_param.shared_init.shape, (1, 1, 128 // 2, 128 // 2))
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_single_shape_bchw_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (1, 3, 128 // 2, 128 // 2)
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertEqual(image_param.shared_init.dim(), 4)
        self.assertEqual(image_param.shared_init.shape, (1, 1, 128 // 2, 128 // 2))
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_multiple_shapes_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (
            (1, 3, 128 // 2, 128 // 2),
            (1, 3, 128 // 4, 128 // 4),
            (1, 3, 128 // 8, 128 // 8),
            (2, 3, 128 // 8, 128 // 8),
            (1, 3, 128 // 16, 128 // 16),
            (2, 3, 128 // 16, 128 // 16),
        )
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertEqual(image_param.shared_init.dim(), 4)
        self.assertEqual(image_param.shared_init.shape, (1, 1, 128 // 2, 128 // 2))
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_multiple_shapes_diff_len_forward(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (
            (128 // 2, 128 // 2),
            (7, 3, 128 // 4, 128 // 4),
            (3, 128 // 8, 128 // 8),
            (2, 4, 128 // 8, 128 // 8),
            (1, 3, 128 // 16, 128 // 16),
            (2, 2, 128 // 16, 128 // 16),
        )
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertEqual(image_param.shared_init.dim(), 4)
        self.assertEqual(image_param.shared_init.shape, (1, 1, 128 // 2, 128 // 2))
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])


if __name__ == "__main__":
    unittest.main()
