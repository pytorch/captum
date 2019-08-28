from __future__ import print_function

import unittest
import torch

from captum._attribution.saliency import Saliency
from captum._attribution.noise_tunnel import NoiseTunnel

from .helpers.basic_models import BasicModel
from .helpers.classification_models import SoftmaxModel


# TODO add more unit tests when the input is a tuple
class Test(unittest.TestCase):
    def test_saliency_test_basic_vanilla(self):
        self._saliency_base_helper()

    def test_saliency_test_basic_smoothgrad(self):
        self._saliency_base_helper(reg_type="smoothgrad")

    def test_saliency_test_basic_vargrad(self):
        self._saliency_base_helper(reg_type="vargrad")

    def test_saliency_classification_vanilla(self):
        self._saliency_classification_helper()

    def test_saliency_classification_smoothgrad(self):
        self._saliency_classification_helper(reg_type="smoothgrad")

    def test_saliency_classification_vargrad(self):
        self._saliency_classification_helper(reg_type="vargrad")

    def _saliency_base_helper(self, reg_type="vanilla"):
        input = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, 7.0], requires_grad=True)
        # manually percomputed gradients
        grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
        model = BasicModel()
        saliency = Saliency(model.forward)
        if reg_type == "vanilla":
            attributions = saliency.attribute(input)
            expected = torch.abs(grads)
            self.assertEqual(
                expected.detach().numpy().tolist(),
                attributions.detach().numpy().tolist(),
            )
        else:
            nt = NoiseTunnel(saliency)
            attributions = nt.attribute(
                input, reg_type=reg_type, n_samples=10, noise_frac=0.0002
            )
        self.assertEqual(input.shape, attributions.shape)

    def _saliency_classification_helper(self, reg_type="vanilla"):
        num_in = 5
        input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        target = torch.tensor(5)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        saliency = Saliency(model.forward)

        if reg_type == "vanilla":
            attributions = saliency.attribute(input, target)

            output = model(input)[:, target]
            output.backward()
            expected = torch.abs(input.grad)
            self.assertEqual(
                expected.detach().numpy().tolist(),
                attributions.detach().numpy().tolist(),
            )
        else:
            nt = NoiseTunnel(saliency)
            attributions = nt.attribute(
                input, reg_type=reg_type, n_samples=10, noise_frac=0.0002, target=target
            )
        self.assertEqual(input.shape, attributions.shape)
