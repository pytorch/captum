#!/usr/bin/env python3
import unittest

import torch

from captum.optim._models.inception_v1 import googlenet
from captum.optim._utils.models import RedirectedReluLayer, ReluLayer
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


def check_layer_not_in_model(self, model, layer) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            check_layer_not_in_model(self, child, layer)


class TestInceptionV1(BaseTest):
    def test_load_inceptionv1_with_redirected_relu(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception"
                + " due to insufficient Torch version."
            )
        try:
            googlenet(pretrained=True, replace_relus_with_redirectedrelu=True)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_load_inceptionv1_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception RedirectedRelu"
                + " due to insufficient Torch version."
            )
        try:
            model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=False)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)
        check_layer_not_in_model(self, model, RedirectedReluLayer)

    def test_load_inceptionv1_linear(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception linear"
                + " due to insufficient Torch version."
            )
        try:
            model = googlenet(pretrained=True, use_linear_modules_only=True)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)
        check_layer_not_in_model(self, model, RedirectedReluLayer)
        check_layer_not_in_model(self, model, ReluLayer)
        check_layer_not_in_model(self, model, torch.nn.MaxPool2d)

    def test_transform_inceptionv1(self) -> None:
        if torch.__version__ <= "1.2.0":
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
        if torch.__version__ <= "1.2.0":
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
        if torch.__version__ <= "1.2.0":
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
        if torch.__version__ <= "1.2.0":
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
