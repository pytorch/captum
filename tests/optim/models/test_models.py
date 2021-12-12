#!/usr/bin/env python3
import unittest
from typing import Type

import torch

from captum.optim.models import googlenet, googlenet_places365
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
                "Skipping load pretrained InceptionV1 test due to insufficient Torch"
                + " version."
            )
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=True)
        _check_layer_in_model(self, model, RedirectedReluLayer)

    def test_load_inceptionv1_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 RedirectedRelu test due to"
                + " insufficient Torch version."
            )
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=False)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_in_model(self, model, torch.nn.ReLU)

    def test_load_inceptionv1_linear(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 linear test due to insufficient"
                + " Torch version."
            )
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_not_in_model(self, model, torch.nn.ReLU)
        _check_layer_not_in_model(self, model, torch.nn.MaxPool2d)
        _check_layer_in_model(self, model, SkipLayer)
        _check_layer_in_model(self, model, torch.nn.AvgPool2d)

    def test_inceptionv1_transform(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping InceptionV1 internal transform test due to insufficient"
                + " Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - 117
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_inceptionv1_transform_warning(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping InceptionV1 internal transform warning test due to"
                + " insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = googlenet(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_inceptionv1_transform_bgr(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping InceptionV1 internal transform BGR test due to insufficient"
                + " Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True, bgr_transform=True)
        output = model._transform_input(x)
        expected_output = x[:, [2, 1, 0]] * 255 - 117
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_inceptionv1_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 forward test due to insufficient"
                + " Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet(pretrained=True)
        outputs = model(x)
        self.assertEqual(list(outputs.shape), [1, 1008])

    def test_inceptionv1_load_and_forward_diff_sizes(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 forward with different sized inputs"
                + " due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 512, 512)
        x2 = torch.zeros(1, 3, 383, 511)
        model = googlenet(pretrained=True)
        outputs = model(x)
        outputs2 = model(x2)
        self.assertEqual(list(outputs.shape), [1, 1008])
        self.assertEqual(list(outputs2.shape), [1, 1008])

    def test_inceptionv1_forward_aux(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 with aux logits forward due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet(pretrained=False, aux_logits=True)
        outputs = model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 1008]] * 3)

    def test_inceptionv1_forward_cuda(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 forward CUDA test due to insufficient"
                + " Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 forward CUDA test due to not"
                + " supporting CUDA."
            )
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = googlenet(pretrained=True).cuda()
        outputs = model(x)
        self.assertTrue(outputs.is_cuda)
        self.assertEqual(list(outputs.shape), [1, 1008])

    def test_inceptionv1_load_and_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 load & JIT test"
                + " due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet(pretrained=True)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual(list(outputs.shape), [1, 1008])

    def test_inceptionv1_load_and_jit_module_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 load & JIT with no"
                + " redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=False)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual(list(outputs.shape), [1, 1008])


class TestInceptionV1Places365(BaseTest):
    def test_load_inceptionv1_places365_with_redirected_relu(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 Places365 due to insufficient"
                + " Torch version."
            )
        model = googlenet_places365(
            pretrained=True, replace_relus_with_redirectedrelu=True
        )
        _check_layer_in_model(self, model, RedirectedReluLayer)

    def test_load_inceptionv1_places365_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 Places365 RedirectedRelu test"
                + " due to insufficient Torch version."
            )
        model = googlenet_places365(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_in_model(self, model, torch.nn.ReLU)

    def test_load_inceptionv1_places365_linear(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 Places365 linear test due to"
                + " insufficient Torch version."
            )
        model = googlenet_places365(pretrained=True, use_linear_modules_only=True)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_not_in_model(self, model, torch.nn.ReLU)
        _check_layer_not_in_model(self, model, torch.nn.MaxPool2d)
        _check_layer_in_model(self, model, SkipLayer)
        _check_layer_in_model(self, model, torch.nn.AvgPool2d)

    def test_inceptionv1_places365_transform(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping InceptionV1 Places365 internal transform test due to"
                + " insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet_places365(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - torch.tensor(
            [116.7894, 112.6004, 104.0437], device=x.device
        ).view(3, 1, 1)
        expected_output = expected_output[:, [2, 1, 0]]
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_inceptionv1_places365_transform_warning(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping InceptionV1 Places365 internal transform warning test due"
                + " to insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = googlenet_places365(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_inceptionv1_places365_load_and_forward(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping basic pretrained InceptionV1 Places365 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(pretrained=True)
        outputs = model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)

    def test_inceptionv1_places365_load_and_forward_diff_sizes(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 forward with different"
                + " sized inputs test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 512, 512)
        x2 = torch.zeros(1, 3, 383, 511)
        model = googlenet_places365(pretrained=True)

        outputs = model(x)
        outputs2 = model(x2)

        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)
        self.assertEqual([list(o.shape) for o in outputs2], [[1, 365]] * 3)

    def test_inceptionv1_places365_forward_no_aux(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 with aux logits forward"
                + " test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(pretrained=False, aux_logits=False)
        outputs = model(x)
        self.assertEqual(list(outputs.shape), [1, 365])

    def test_inceptionv1_places365_forward_cuda(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 forward CUDA test due to"
                + " insufficient Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 forward CUDA test due to"
                + " not supporting CUDA."
            )
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = googlenet_places365(pretrained=True).cuda()
        outputs = model(x)

        self.assertTrue(outputs[0].is_cuda)
        self.assertTrue(outputs[1].is_cuda)
        self.assertTrue(outputs[2].is_cuda)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)

    def test_inceptionv1_places365_load_and_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 load & JIT module test"
                + " due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(pretrained=True)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)

    def test_inceptionv1_places365_jit_module_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 load & JIT module with no"
                + " redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)
