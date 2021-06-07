#!/usr/bin/env python3
import unittest
from typing import Type

import torch

from captum.optim.models import googlenet, vgg16
from captum.optim.models._common import RedirectedReluLayer, SkipLayer
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


def _check_layer_in_model(
    self,
    model: torch.nn.Module,
    layer: Type[torch.nn.Module],
) -> None:
    def check_for_layer_in_model(model, layer) -> bool:
        for name, child in model._modules.items():
            if child is not None:
                if isinstance(child, layer):
                    return True
                if check_for_layer_in_model(child, layer):
                    return True
        return False

    self.assertTrue(check_for_layer_in_model(model, layer))


def _check_layer_not_in_model(
    self, model: torch.nn.Module, layer: Type[torch.nn.Module]
) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            _check_layer_not_in_model(self, child, layer)


class TestInceptionV1(BaseTest):
    def test_load_inceptionv1_with_redirected_relu(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception"
                + " due to insufficient Torch version."
            )
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=True)
        _check_layer_in_model(self, model, RedirectedReluLayer)

    def test_load_inceptionv1_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception RedirectedRelu"
                + " due to insufficient Torch version."
            )
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=False)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_in_model(self, model, torch.nn.ReLU)

    def test_load_inceptionv1_linear(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained inception linear"
                + " due to insufficient Torch version."
            )
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_not_in_model(self, model, torch.nn.ReLU)
        _check_layer_not_in_model(self, model, torch.nn.MaxPool2d)
        _check_layer_in_model(self, model, SkipLayer)
        _check_layer_in_model(self, model, torch.nn.AvgPool2d)

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


class TestVGG16(BaseTest):
    def test_load_vgg16_with_redirected_relu(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained VGG-16 due to insufficient"
                + " Torch version."
            )
        model = vgg16(pretrained=True, replace_relus_with_redirectedrelu=True)
        _check_layer_in_model(self, model, RedirectedReluLayer)

    def test_load_vgg16_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained VGG-16 RedirectedRelu test"
                + " due to insufficient Torch version."
            )
        model = vgg16(pretrained=True, replace_relus_with_redirectedrelu=False)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_in_model(self, model, torch.nn.ReLU)

    def test_load_vgg16_linear(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained VGG-16 linear test due to"
                + " insufficient Torch version."
            )
        model = vgg16(pretrained=True, use_linear_modules_only=True)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_not_in_model(self, model, torch.nn.ReLU)
        _check_layer_not_in_model(self, model, torch.nn.MaxPool2d)
        _check_layer_in_model(self, model, SkipLayer)
        _check_layer_in_model(self, model, torch.nn.AvgPool2d)

    def test_vgg16_transform(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping VGG-16 internal transform test due to"
                + " insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = vgg16(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - torch.tensor(
            [123.68, 116.779, 103.939], device=x.device
        ).view(3, 1, 1)
        expected_output = expected_output[:, [2, 1, 0]]
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_vgg16_transform_no_scale(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping VGG-16 internal transform test due to"
                + " insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1) * 255
        model = vgg16(pretrained=True, scale_input=False)
        output = model._transform_input(x)
        expected_output = x - torch.tensor(
            [123.68, 116.779, 103.939], device=x.device
        ).view(3, 1, 1)
        expected_output = expected_output[:, [2, 1, 0]]
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_vgg16_transform_warning(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping VGG-16 internal transform warning test due"
                + " to insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = vgg16(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_vgg16_load_and_forward(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping basic pretrained VGG-16 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = vgg16(pretrained=True)
        outputs = model(x)
        self.assertEqual(list(outputs.shape), [1, 512, 7, 7])

    def test_vgg16_load_and_forward_diff_sizes(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained VGG-16 forward with different"
                + " sized inputs test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 512, 512)
        x2 = torch.zeros(1, 3, 383, 511)
        model = vgg16(pretrained=True)

        outputs = model(x)
        outputs2 = model(x2)

        self.assertEqual(list(outputs.shape), [1, 512, 7, 7])
        self.assertEqual(list(outputs2.shape), [1, 512, 7, 7])

    def test_vgg16_forward_classifier(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained VGG-16 with classifier_logits forward"
                + " test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = vgg16(pretrained=False, classifier_logits=True)
        outputs = model(x)
        self.assertEqual(list(outputs.shape), [1, 1000])

    def test_vgg16_forward_cuda(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained VGG-16 forward CUDA test due to"
                + " insufficient Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained VGG-16 forward CUDA test due to"
                + " not supporting CUDA."
            )
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = vgg16(pretrained=True).cuda()
        output = model(x)

        self.assertTrue(output.is_cuda)
        self.assertEqual(list(output.shape), [1, 512, 7, 7])
