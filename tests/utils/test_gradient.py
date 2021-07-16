#!/usr/bin/env python3

from typing import List, Tuple

import torch
import torch.nn as nn

from captum._utils.gradient import (
    _compute_jacobian_wrt_params,
    _compute_jacobian_wrt_params_autograd_hacks,
    apply_gradient_requirements,
    compute_gradients,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.helpers.basic_models import (
    BasicLinearModel2,
    BasicLinearModel_Multilayer,
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
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [0.0], delta=0.01)
        # Verify grad attribute is not altered
        assertArraysAlmostEqual(input.grad.squeeze(0).tolist(), [9.0], delta=0.0)

    def test_gradient_basic_2(self) -> None:
        model = BasicModel()
        input = torch.tensor([[-3.0]], requires_grad=True)
        input.grad = torch.tensor([[14.0]])
        grads = compute_gradients(model, input)[0]
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [1.0], delta=0.01)
        # Verify grad attribute is not altered
        assertArraysAlmostEqual(input.grad.squeeze(0).tolist(), [14.0], delta=0.0)

    def test_gradient_multiinput(self) -> None:
        model = BasicModel6_MultiTensor()
        input1 = torch.tensor([[-3.0, -5.0]], requires_grad=True)
        input2 = torch.tensor([[-5.0, 2.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2))
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)
        assertArraysAlmostEqual(grads[1].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)

    def test_gradient_additional_args(self) -> None:
        model = BasicModel4_MultiArgs()
        input1 = torch.tensor([[10.0]], requires_grad=True)
        input2 = torch.tensor([[8.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2), additional_forward_args=(2,))
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [1.0], delta=0.01)
        assertArraysAlmostEqual(grads[1].squeeze(0).tolist(), [-0.5], delta=0.01)

    def test_gradient_additional_args_2(self) -> None:
        model = BasicModel5_MultiArgs()
        input1 = torch.tensor([[-10.0]], requires_grad=True)
        input2 = torch.tensor([[6.0]], requires_grad=True)
        grads = compute_gradients(
            model, (input1, input2), additional_forward_args=([3, -4],)
        )
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [0.0], delta=0.01)
        assertArraysAlmostEqual(grads[1].squeeze(0).tolist(), [4.0], delta=0.01)

    def test_gradient_target_int(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, 5.0]], requires_grad=True)
        grads0 = compute_gradients(model, (input1, input2), target_ind=0)
        grads1 = compute_gradients(model, (input1, input2), target_ind=1)
        assertArraysAlmostEqual(grads0[0].squeeze(0).tolist(), [1.0, 0.0], delta=0.01)
        assertArraysAlmostEqual(grads0[1].squeeze(0).tolist(), [-1.0, 0.0], delta=0.01)
        assertArraysAlmostEqual(grads1[0].squeeze(0).tolist(), [0.0, 0.0], delta=0.01)
        assertArraysAlmostEqual(grads1[1].squeeze(0).tolist(), [0.0, 0.0], delta=0.01)

    def test_gradient_target_list(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2), target_ind=[0, 1])
        assertArraysAlmostEqual(
            torch.flatten(grads[0]).tolist(), [1.0, 0.0, 0.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            torch.flatten(grads[1]).tolist(), [-1.0, 0.0, 0.0, -1.0], delta=0.01
        )

    def test_gradient_target_tuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        grads = compute_gradients(model, input, target_ind=(0, 1))[0]
        assertArraysAlmostEqual(
            torch.flatten(grads).tolist(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            delta=0.01,
        )

    def test_gradient_target_listtuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        target: List[Tuple[int, ...]] = [(1, 1), (0, 1)]
        grads = compute_gradients(model, input, target_ind=target)[0]
        assertArraysAlmostEqual(
            torch.flatten(grads).tolist(),
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            delta=0.01,
        )

    def test_gradient_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[1.0, 6.0, -3.0]], requires_grad=True)
        grads = compute_gradients(model, input, target_ind=0)[0]
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [3.0, 3.0, 3.0], delta=0.01)

    def test_layer_gradient_linear0(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, -11.0, 23.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear0, input, target_ind=0
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [4.0, 4.0, 4.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [5.0, -11.0, 23.0], delta=0.01
        )

    def test_layer_gradient_linear1(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )

    def test_layer_gradient_linear1_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )

    def test_layer_gradient_relu_input_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.relu, input, target_ind=1, attribute_to_layer_input=True
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )

    def test_layer_gradient_output(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear2, input, target_ind=1
        )
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)
        assertArraysAlmostEqual(eval[0].squeeze(0).tolist(), [26.0, 28.0], delta=0.01)

    def test_jacobian_scores_single_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float())

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], a)

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], a)

    def test_jacobian_scores_single_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], a)

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], a)

    def test_jacobian_scores_single_scalar_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 1)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(1, 3).view(1, 2).float())

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a, 2 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([10, 35]))

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a, 2 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([10, 35]))

    def test_jacobian_scores_single_vector_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 2)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(0, 4).view(2, 2).float())

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((2 * a, 4 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((2 * a, 4 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))

    def test_jacobian_scores_batch_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], a[0])
        assertTensorAlmostEqual(self, grads[0][1], a[1])

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], a[0])
        assertTensorAlmostEqual(self, grads[0][1], a[1])

    def test_jacobian_scores_batch_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], a[0])))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], a[1])))

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], a[0])))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], a[1])))

    def test_jacobian_scores_batch_scalar_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 1)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(1, 3).view(1, 2).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], 2 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([10, 35]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], 2 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([20, 70]))

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], 2 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([10, 35]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], 2 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([20, 70]))

    def test_jacobian_scores_batch_vector_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 2)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(0, 4).view(2, 2).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((2 * a[0], 4 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((2 * a[1], 4 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([[20, 70], [20, 70]]))

        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a)
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((2 * a[0], 4 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((2 * a[1], 4 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([[20, 70], [20, 70]]))

    def test_jacobian_loss_single_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float())

        a = torch.ones(5).unsqueeze(0)
        label = torch.Tensor([9])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, a, label, loss_fn)
        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a)

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
        assertTensorAlmostEqual(self, grads[0], 2 * (10 - 9) * a)

    def test_jacobian_loss_single_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.ones(5).unsqueeze(0)
        label = torch.Tensor([[9, 38]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a, 2 * (35 - 38) * a))
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0], torch.stack((2 * (10 - 9) * a, 2 * (35 - 38) * a))
        )

    def test_jacobian_loss_batch_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([9, 18])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, a, label, loss_fn)
        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a[0])
        assertTensorAlmostEqual(self, grads[0][1], 2 * (20 - 18) * a[1])

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a[0])
        assertTensorAlmostEqual(self, grads[0][1], 2 * (20 - 18) * a[1])

    def test_jacobian_loss_batch_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

    def test_jacobian_loss_single_scalar_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 1)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(1, 3).view(1, 2).float())

        a = torch.ones(5).unsqueeze(0)
        label = torch.Tensor([[78]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (80 - 78) * a, 2 * 2 * (80 - 78) * a))
        )
        assertTensorAlmostEqual(
            self, grads[1][0], 2 * (80 - 78) * torch.Tensor([10, 35])
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (80 - 78) * a, 2 * 2 * (80 - 78) * a))
        )
        assertTensorAlmostEqual(
            self, grads[1][0], 2 * (80 - 78) * torch.Tensor([10, 35])
        )

    def test_jacobian_loss_batch_vector_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 2)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(0, 4).view(2, 2).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[33, 124], [69, 256]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, a, label, loss_fn)
        assertTensorAlmostEqual(
            self,
            grads[0][0],
            torch.stack(
                (
                    2 * (0 * (35 - 33) + 2 * (125 - 124)) * a[0],
                    2 * (1 * (35 - 33) + 3 * (125 - 124)) * a[0],
                )
            ),
        )
        assertTensorAlmostEqual(
            self,
            grads[1][0],
            torch.Tensor(
                [
                    [2 * (35 - 33) * 10, 2 * (35 - 33) * 35],
                    [2 * (125 - 124) * 10, 2 * (125 - 124) * 35],
                ]
            ),
        )
        assertTensorAlmostEqual(
            self,
            grads[0][1],
            torch.stack(
                (
                    2 * (0 * (70 - 69) + 2 * (250 - 256)) * a[1],
                    2 * (1 * (70 - 69) + 3 * (250 - 256)) * a[1],
                )
            ),
        )
        assertTensorAlmostEqual(
            self,
            grads[1][1],
            torch.Tensor(
                [
                    [2 * (70 - 69) * 10 * 2, 2 * (70 - 69) * 35 * 2],
                    [2 * (250 - 256) * 10 * 2, 2 * (250 - 256) * 35 * 2],
                ]
            ),
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads_h = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
        assertTensorAlmostEqual(self, grads_h[0][0], grads[0][0])
        assertTensorAlmostEqual(self, grads_h[1][0], grads[1][0])
        assertTensorAlmostEqual(self, grads_h[0][1], grads[0][1])
        assertTensorAlmostEqual(self, grads_h[1][1], grads[1][1])

    def test_jacobian_loss_custom_correct(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return torch.square(out - label)

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])
        grads = _compute_jacobian_wrt_params(model, a, label, my_loss)

        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

    def test_jacobian_loss_custom_wrong(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return torch.sum(torch.square(out - label))

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        with self.assertRaises(AssertionError):
            _compute_jacobian_wrt_params(model, a, label, my_loss)

    def test_jacobian_loss_custom_correct_hack(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return torch.sum(torch.square(out - label))

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])
        grads = _compute_jacobian_wrt_params_autograd_hacks(model, a, label, my_loss)

        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

    def test_jacobian_loss_custom_wrong_hack(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return torch.square(out - label)

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        with self.assertRaises(AssertionError):
            _compute_jacobian_wrt_params_autograd_hacks(model, a, label, my_loss)

    def test_jacobian_loss_wrong_reduction_hack(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        loss_fn = nn.MSELoss(reduction="none")

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        with self.assertRaises(AssertionError):
            _compute_jacobian_wrt_params_autograd_hacks(model, a, label, loss_fn)
