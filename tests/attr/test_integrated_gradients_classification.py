from __future__ import print_function

import torch
import unittest

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel

from .helpers.classification_models import SigmoidModel, SoftmaxModel


class Test(unittest.TestCase):
    def test_sigmoid_classification_vanilla(self):
        self._assert_sigmoid_classification("vanilla")

    def test_sigmoid_classification_smoothgrad(self):
        self._assert_sigmoid_classification("smoothgrad")

    def test_sigmoid_classification_vargrad(self):
        self._assert_sigmoid_classification("vargrad")

    def test_softmax_classification_vanilla(self):
        self._assert_softmax_classification("vanilla")

    def test_softmax_classification_smoothgrad(self):
        self._assert_softmax_classification("smoothgrad")

    def test_softmax_classification_vargrad(self):
        self._assert_softmax_classification("vargrad")

    def _assert_sigmoid_classification(self, type="vanilla"):
        num_in = 20
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        target = torch.tensor(0)
        # TODO add test cases for multiple different layers
        model = SigmoidModel(num_in, 5, 1)
        self._validate_completness(model, input, target, type)

    def _assert_softmax_classification(self, type="vanilla"):
        num_in = 40
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        target = torch.tensor(5)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        self._validate_completness(model, input, target, type)

    def _validate_completness(self, model, inputs, target, type="vanilla"):
        ig = IntegratedGradients(model.forward)
        for method in [
            "riemann_right",
            "riemann_left",
            "riemann_middle",
            "riemann_trapezoid",
            "gausslegendre",
        ]:
            model.zero_grad()
            if type == "vanilla":
                attributions, delta = ig.attribute(
                    inputs, target=target, method=method, n_steps=1000
                )
                # attributions are returned as tuples for the integrated_gradients
                self.assertAlmostEqual(
                    attributions.sum(),
                    model.forward(inputs)[:, target]
                    - model.forward(0 * inputs)[:, target],
                    delta=0.005,
                )
                delta_expected = abs(
                    attributions.sum().item()
                    - (
                        model.forward(inputs)[:, target].item()
                        - model.forward(0 * inputs)[:, target].item()
                    )
                )
                self.assertAlmostEqual(delta, delta_expected, delta=0.005)
            else:
                nt = NoiseTunnel(ig)
                attributions, delta = nt.attribute(
                    inputs,
                    reg_type=type,
                    n_samples=10,
                    noise_frac=0.0002,
                    target=target,
                    method=method,
                )
            self.assertEqual(attributions.shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()
