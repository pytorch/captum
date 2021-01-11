#!/usr/bin/env python3
import unittest

import torch

from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestInceptionV1(BaseTest):
    def test_load_inceptionv1(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception"
                + " due to insufficient Torch version."
            )
        try:
            googlenet(pretrained=True)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_transform_inceptionv1(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping inceptionV1 internal transform"
                + " due to insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - 117
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_transform_bgr_inceptionv1(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping inceptionV1 internal transform"
                + " BGR due to insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True, bgr_transform=True)
        output = model._transform_input(x)
        expected_output = x[:, [2, 1, 0]] * 255 - 117
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_load_and_forward_basic_inceptionv1(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping basic pretrained inceptionV1 forward"
                + " due to insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True)
        try:
            model(x)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_load_and_forward_diff_sizes_inceptionv1(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping pretrained inceptionV1 forward with different sized inputs"
                + " due to insufficient Torch version."
            )
        x = torch.randn(1, 3, 512, 512).clamp(0, 1)
        x2 = torch.randn(1, 3, 383, 511).clamp(0, 1)
        model = googlenet(pretrained=True)
        try:
            model(x)
            model(x2)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_forward_aux_inceptionv1(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping pretrained inceptionV1 with aux logits forward"
                + " due to insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=False, aux_logits=True)
        outputs = model(x)
        self.assertEqual(len(outputs), 3)


if __name__ == "__main__":
    unittest.main()
