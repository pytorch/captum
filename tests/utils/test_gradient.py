#!/usr/bin/env python3

from typing import List, Tuple

import torch
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel5_MultiArgs,
    BasicModel6_MultiTensor,
    BasicModel_MultiLayer,
)


class Test(BaseTest):
    def test_apply_gradient_reqs(self) -> None:
        initial_grads = [False, True, False]
        test_tensor = torch.tensor([[6.0]], requires_grad=True)
        test_tensor.grad = torch.tensor([[7.0]])
        test_tensor_tuple = (torch.tensor([[5.0]]), test_tensor, torch.tensor([[7.0]]))
        out_mask = apply_gradient_requirements(test_tensor_tuple)
        for i in range(len(test_tensor_tuple)):
            self.assertTrue(test_tensor_tuple[i].requires_grad)
            self.assertEqual(out_mask[i], initial_grads[i])

    def test_undo_gradient_reqs(self) -> None:
        initial_grads = [False, True, False]
        test_tensor = torch.tensor([[6.0]], requires_grad=True)
        test_tensor.grad = torch.tensor([[7.0]])
        test_tensor_tuple = (
            torch.tensor([[6.0]], requires_grad=True),
            test_tensor,
            torch.tensor([[7.0]], requires_grad=True),
        )
        undo_gradient_requirements(test_tensor_tuple, initial_grads)
        for i in range(len(test_tensor_tuple)):
            self.assertEqual(test_tensor_tuple[i].requires_grad, initial_grads[i])

    def test_gradient_basic(self) -> None:
        model = BasicModel()
        input = torch.tensor([[5.0]], requires_grad=True)
        input.grad = torch.tensor([[9.0]])
        grads = compute_gradients(model, input)[0]
        assertTensorAlmostEqual(self, grads, [[0.0]], delta=0.01, mode="max")
        # Verify grad attribute is not altered
        assertTensorAlmostEqual(self, input.grad, [[9.0]], delta=0.0, mode="max")

    def test_gradient_basic_2(self) -> None:
        model = BasicModel()
        input = torch.tensor([[-3.0]], requires_grad=True)
        input.grad = torch.tensor([[14.0]])
        grads = compute_gradients(model, input)[0]
        assertTensorAlmostEqual(self, grads, [[1.0]], delta=0.01, mode="max")
        # Verify grad attribute is not altered
        assertTensorAlmostEqual(self, input.grad, [[14.0]], delta=0.0, mode="max")

    def test_gradient_multiinput(self) -> None:
        model = BasicModel6_MultiTensor()
        input1 = torch.tensor([[-3.0, -5.0]], requires_grad=True)
        input2 = torch.tensor([[-5.0, 2.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2))
        assertTensorAlmostEqual(self, grads[0], [[0.0, 1.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads[1], [[0.0, 1.0]], delta=0.01, mode="max")

    def test_gradient_additional_args(self) -> None:
        model = BasicModel4_MultiArgs()
        input1 = torch.tensor([[10.0]], requires_grad=True)
        input2 = torch.tensor([[8.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2), additional_forward_args=(2,))
        assertTensorAlmostEqual(self, grads[0], [[1.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads[1], [[-0.5]], delta=0.01, mode="max")

    def test_gradient_additional_args_2(self) -> None:
        model = BasicModel5_MultiArgs()
        input1 = torch.tensor([[-10.0]], requires_grad=True)
        input2 = torch.tensor([[6.0]], requires_grad=True)
        grads = compute_gradients(
            model, (input1, input2), additional_forward_args=([3, -4],)
        )
        assertTensorAlmostEqual(self, grads[0], [[0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads[1], [[4.0]], delta=0.01, mode="max")

    def test_gradient_target_int(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, 5.0]], requires_grad=True)
        grads0 = compute_gradients(model, (input1, input2), target_ind=0)
        grads1 = compute_gradients(model, (input1, input2), target_ind=1)
        assertTensorAlmostEqual(self, grads0[0], [[1.0, 0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads0[1], [[-1.0, 0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads1[0], [[0.0, 0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads1[1], [[0.0, 0.0]], delta=0.01, mode="max")

    def test_gradient_target_list(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2), target_ind=[0, 1])
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[1.0, 0.0], [0.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            grads[1],
            [[-1.0, 0.0], [0.0, -1.0]],
            delta=0.01,
            mode="max",
        )

    def test_gradient_target_tuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        grads = compute_gradients(model, input, target_ind=(0, 1))[0]
        assertTensorAlmostEqual(
            self,
            grads,
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            delta=0.01,
            mode="max",
        )

    def test_gradient_target_listtuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        target: List[Tuple[int, ...]] = [(1, 1), (0, 1)]
        grads = compute_gradients(model, input, target_ind=target)[0]
        assertTensorAlmostEqual(
            self,
            grads,
            [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]],
            delta=0.01,
            mode="max",
        )

    def test_gradient_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[1.0, 6.0, -3.0]], requires_grad=True)
        grads = compute_gradients(model, input, target_ind=0)[0]
        assertTensorAlmostEqual(self, grads, [[3.0, 3.0, 3.0]], delta=0.01, mode="max")

    def test_layer_gradient_linear0(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, -11.0, 23.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear0, input, target_ind=0
        )
        assertTensorAlmostEqual(
            self, grads[0], [[4.0, 4.0, 4.0]], delta=0.01, mode="max"
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[5.0, -11.0, 23.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_linear1(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[0.0, 1.0, 1.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[-2.0, 9.0, 9.0, 9.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_linear1_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[0.0, 1.0, 1.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[-2.0, 9.0, 9.0, 9.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_relu_input_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.relu, input, target_ind=1, attribute_to_layer_input=True
        )
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[0.0, 1.0, 1.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[-2.0, 9.0, 9.0, 9.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_output(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear2, input, target_ind=1
        )
        assertTensorAlmostEqual(self, grads[0], [[0.0, 1.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, eval[0], [[26.0, 28.0]], delta=0.01, mode="max")
