from __future__ import print_function

import unittest
import torch

from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.noise_tunnel import NoiseTunnel

from .helpers.basic_models import BasicModel
from .helpers.classification_models import SoftmaxModel


# TODO add more unit tests when the input is a tuple
class Test(unittest.TestCase):
    def test_input_x_gradient_test_basic_vanilla(self):
        self._input_x_gradient_base_helper()

    def test_input_x_gradient_test_basic_smoothgrad(self):
        self._input_x_gradient_base_helper(reg_type="smoothgrad")

    def test_input_x_gradient_test_basic_vargrad(self):
        self._input_x_gradient_base_helper(reg_type="vargrad")

    def test_input_x_gradient_classification_vanilla(self):
        self._input_x_gradient_classification_helper()

    def test_input_x_gradient_classification_smoothgrad(self):
        self._input_x_gradient_classification_helper(reg_type="smoothgrad")

    def test_input_x_gradient_classification_vargrad(self):
        self._input_x_gradient_classification_helper(reg_type="vargrad")

    def _input_x_gradient_classification_helper(self, reg_type="vanilla"):
        num_in = 5
        input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        target = torch.tensor(5)

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        input_x_grad = InputXGradient(model.forward)
        if reg_type == "vanilla":
            attributions = input_x_grad.attribute(input, target)
            output = model(input)[:, target]
            output.backward()
            expercted = input.grad * input
            self.assertEqual(
                expercted.detach().numpy().tolist(),
                attributions.detach().numpy().tolist(),
            )
        else:
            nt = NoiseTunnel(input_x_grad)
            attributions = nt.attribute(
                input, reg_type=reg_type, n_samples=10, noise_frac=0.0002, target=target
            )

        self.assertAlmostEqual(attributions.shape, input.shape)

    def _input_x_gradient_base_helper(self, reg_type="vanilla"):
        input = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, 7.0], requires_grad=True)
        # manually percomputed gradients
        grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
        model = BasicModel()
        input_x_grad = InputXGradient(model.forward)
        if reg_type == "vanilla":
            attributions = input_x_grad.attribute(input)
            expected = grads * input
            self.assertEqual(
                expected.detach().numpy().tolist(),
                attributions.detach().numpy().tolist(),
            )
        else:
            nt = NoiseTunnel(input_x_grad)
            attributions = nt.attribute(
                input, reg_type=reg_type, n_samples=10, noise_frac=0.0002
            )
        self.assertAlmostEqual(attributions.shape, input.shape)
