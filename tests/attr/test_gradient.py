from __future__ import print_function

import torch

from captum.attr._utils.gradient import (
    compute_gradients,
    compute_layer_gradients_and_eval,
)

from .helpers.utils import assertArraysAlmostEqual, BaseTest
from .helpers.basic_models import (
    BasicModel,
    BasicModel6_MultiTensor,
    TestModel_MultiLayer,
)


class Test(BaseTest):
    def test_gradient_basic(self):
        model = BasicModel()
        input = torch.tensor([[5.0]], requires_grad=True)
        grads = compute_gradients(model, input)[0]
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [0.0], delta=0.01)

    def test_gradient_basic_2(self):
        model = BasicModel()
        input = torch.tensor([[-3.0]], requires_grad=True)
        grads = compute_gradients(model, input)[0]
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [1.0], delta=0.01)

    def test_gradient_multiinput(self):
        model = BasicModel6_MultiTensor()
        input1 = torch.tensor([[-3.0, -5.0]], requires_grad=True)
        input2 = torch.tensor([[-5.0, 2.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2))
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)
        assertArraysAlmostEqual(grads[1].squeeze(0).tolist(), [0.0, 0.0], delta=0.01)

    def test_layer_gradient_linear0(self):
        model = TestModel_MultiLayer()
        input = torch.tensor([[5.0, -11.0, 23.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear0, input, target_ind=0
        )
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [4.0, 4.0, 4.0], delta=0.01)
        assertArraysAlmostEqual(
            eval.squeeze(0).tolist(), [5.0, -11.0, 23.0], delta=0.01
        )

    def test_layer_gradient_linear1(self):
        model = TestModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertArraysAlmostEqual(
            grads.squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval.squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )

    def test_layer_gradient_output(self):
        model = TestModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear2, input, target_ind=1
        )
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [0.0, 1.0], delta=0.01)
        assertArraysAlmostEqual(eval.squeeze(0).tolist(), [26.0, 28.0], delta=0.01)
