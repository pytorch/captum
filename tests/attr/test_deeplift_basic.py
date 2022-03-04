#!/usr/bin/env python3

from inspect import signature
from typing import Callable, List, Tuple, Union

import torch
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.integrated_gradients import IntegratedGradients
from tests.helpers.basic import (
    BaseTest,
    assertAttributionComparision,
    assertTensorAlmostEqual,
)
from tests.helpers.basic_models import (
    BasicModelWithReusableModules,
    Conv1dSeqModel,
    LinearMaxPoolLinearModel,
    ReLUDeepLiftModel,
    ReLULinearModel,
    TanhDeepLiftModel,
)
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_relu_deeplift(self) -> None:
        x1 = torch.tensor([1.0], requires_grad=True)
        x2 = torch.tensor([2.0], requires_grad=True)

        b1 = torch.tensor([0.0], requires_grad=True)
        b2 = torch.tensor([0.0], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_deeplift_exact_match(self) -> None:
        x1 = torch.tensor([1.0], requires_grad=True)
        x2 = torch.tensor([2.0], requires_grad=True)

        b1 = torch.tensor([0.0], requires_grad=True)
        b2 = torch.tensor([0.0], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)
        model = ReLUDeepLiftModel()
        dl = DeepLift(model)
        attributions, delta = dl.attribute(
            inputs, baselines, return_convergence_delta=True
        )
        self.assertEqual(attributions[0][0], 2.0)
        self.assertEqual(attributions[1][0], 1.0)
        self.assertEqual(delta[0], 0.0)

    def test_relu_deeplift_exact_match_wo_mutliplying_by_inputs(self) -> None:
        x1 = torch.tensor([1.0])
        x2 = torch.tensor([2.0])
        inputs = (x1, x2)

        model = ReLUDeepLiftModel()
        dl = DeepLift(model, multiply_by_inputs=False)
        attributions = dl.attribute(inputs)
        self.assertEqual(attributions[0][0], 2.0)
        self.assertEqual(attributions[1][0], 0.5)

    def test_tanh_deeplift(self) -> None:
        x1 = torch.tensor([-1.0], requires_grad=True)
        x2 = torch.tensor([-2.0], requires_grad=True)

        b1 = torch.tensor([0.0], requires_grad=True)
        b2 = torch.tensor([0.0], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = TanhDeepLiftModel()
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_deeplift_batch(self) -> None:
        x1 = torch.tensor([[1.0], [1.0], [1.0], [1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0], [2.0], [2.0], [2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_linear_deeplift(self) -> None:
        model = ReLULinearModel(inplace=False)
        x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (0, 0.0001)

        # expected = [[[0.0, 0.0]], [[6.0, 2.0]]]
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_linear_deeplift_batch(self) -> None:
        model = ReLULinearModel(inplace=False)
        x1 = torch.tensor([[-10.0, 1.0, -5.0], [2.0, 3.0, 4.0]], requires_grad=True)
        x2 = torch.tensor([[3.0, 3.0, 1.0], [2.3, 5.0, 4.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (torch.zeros(1, 3), torch.rand(1, 3) * 0.001)
        # expected = [[[0.0, 0.0]], [[6.0, 2.0]]]
        self._deeplift_assert(model, DeepLift(model), inputs, baselines)

    def test_relu_deeplift_with_hypothetical_contrib_func(self) -> None:
        model = Conv1dSeqModel()
        rand_seq_data = torch.abs(torch.randn(2, 4, 1000))
        rand_seq_ref = torch.abs(torch.randn(2, 4, 1000))
        dls = DeepLift(model)
        attr = dls.attribute(
            rand_seq_data,
            rand_seq_ref,
            custom_attribution_func=_hypothetical_contrib_func,
            target=(1, 0),
        )
        self.assertEqual(attr.shape, rand_seq_data.shape)

    def test_relu_deepliftshap_batch_4D_input(self) -> None:
        x1 = torch.ones(4, 1, 1, 1)
        x2 = torch.tensor([[[[2.0]]]] * 4)

        b1 = torch.zeros(4, 1, 1, 1)
        b2 = torch.zeros(4, 1, 1, 1)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLiftShap(model), inputs, baselines)

    def test_relu_deepliftshap_batch_4D_input_wo_mutliplying_by_inputs(self) -> None:
        x1 = torch.ones(4, 1, 1, 1)
        x2 = torch.tensor([[[[2.0]]]] * 4)

        b1 = torch.zeros(4, 1, 1, 1)
        b2 = torch.zeros(4, 1, 1, 1)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        attr = DeepLiftShap(model, multiply_by_inputs=False).attribute(
            inputs, baselines
        )
        assertTensorAlmostEqual(self, attr[0], 2 * torch.ones(4, 1, 1, 1))
        assertTensorAlmostEqual(self, attr[1], 0.5 * torch.ones(4, 1, 1, 1))

    def test_relu_deepliftshap_multi_ref(self) -> None:
        x1 = torch.tensor([[1.0]], requires_grad=True)
        x2 = torch.tensor([[2.0]], requires_grad=True)

        b1 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        b2 = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)

        inputs = (x1, x2)
        baselines = (b1, b2)

        model = ReLUDeepLiftModel()
        self._deeplift_assert(model, DeepLiftShap(model), inputs, baselines)

    def test_relu_deepliftshap_baselines_as_func(self) -> None:
        model = ReLULinearModel(inplace=False)
        x1 = torch.tensor([[-10.0, 1.0, -5.0]])
        x2 = torch.tensor([[3.0, 3.0, 1.0]])

        def gen_baselines() -> Tuple[Tensor, ...]:
            b1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            b2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            return (b1, b2)

        def gen_baselines_scalar() -> Tuple[float, ...]:
            return (0.0, 0.0001)

        def gen_baselines_with_inputs(inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
            b1 = torch.cat([inputs[0], inputs[0] - 10])
            b2 = torch.cat([inputs[1], inputs[1] - 10])
            return (b1, b2)

        def gen_baselines_returns_array() -> Tuple[List[List[float]], ...]:
            b1 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            b2 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            return (b1, b2)

        inputs = (x1, x2)

        dl_shap = DeepLiftShap(model)
        self._deeplift_assert(model, dl_shap, inputs, gen_baselines)
        self._deeplift_assert(model, dl_shap, inputs, gen_baselines_with_inputs)
        with self.assertRaises(AssertionError):
            self._deeplift_assert(
                model, DeepLiftShap(model), inputs, gen_baselines_returns_array
            )
        with self.assertRaises(AssertionError):
            self._deeplift_assert(model, dl_shap, inputs, gen_baselines_scalar)

        baselines = gen_baselines()
        attributions = dl_shap.attribute(inputs, baselines)
        attributions_with_func = dl_shap.attribute(inputs, gen_baselines)
        assertTensorAlmostEqual(self, attributions[0], attributions_with_func[0])
        assertTensorAlmostEqual(self, attributions[1], attributions_with_func[1])

    def test_relu_deepliftshap_with_custom_attr_func(self) -> None:
        def custom_attr_func(
            multipliers: Tuple[Tensor, ...],
            inputs: Tuple[Tensor, ...],
            baselines: Tuple[Tensor, ...],
        ) -> Tuple[Tensor, ...]:
            return tuple(multiplier * 0.0 for multiplier in multipliers)

        model = ReLULinearModel(inplace=False)
        x1 = torch.tensor([[-10.0, 1.0, -5.0]])
        x2 = torch.tensor([[3.0, 3.0, 1.0]])
        b1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        b2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        inputs = (x1, x2)
        baselines = (b1, b2)
        dls = DeepLiftShap(model)
        attr_w_func = dls.attribute(
            inputs, baselines, custom_attribution_func=custom_attr_func
        )

        assertTensorAlmostEqual(self, attr_w_func[0], [[0.0, 0.0, 0.0]], 0.0)
        assertTensorAlmostEqual(self, attr_w_func[1], [[0.0, 0.0, 0.0]], 0.0)

    def test_relu_deepliftshap_with_hypothetical_contrib_func(self) -> None:
        model = Conv1dSeqModel()
        rand_seq_data = torch.abs(torch.randn(2, 4, 1000))
        rand_seq_ref = torch.abs(torch.randn(3, 4, 1000))
        dls = DeepLiftShap(model)
        attr = dls.attribute(
            rand_seq_data,
            rand_seq_ref,
            custom_attribution_func=_hypothetical_contrib_func,
            target=(0, 0),
        )
        self.assertEqual(attr.shape, rand_seq_data.shape)

    def test_reusable_modules(self) -> None:
        model = BasicModelWithReusableModules()
        input = torch.rand(1, 3)
        dl = DeepLift(model)
        with self.assertRaises(RuntimeError):
            dl.attribute(input, target=0)

    def test_lin_maxpool_lin_classification(self) -> None:
        inputs = torch.ones(2, 4)
        baselines = torch.tensor([[1, 2, 3, 9], [4, 8, 6, 7]]).float()

        model = LinearMaxPoolLinearModel()
        dl = DeepLift(model)
        attrs, delta = dl.attribute(
            inputs, baselines, target=0, return_convergence_delta=True
        )
        expected = torch.Tensor([[0.0, 0.0, 0.0, -8.0], [0.0, -7.0, 0.0, 0.0]])
        expected_delta = torch.Tensor([0.0, 0.0])
        assertTensorAlmostEqual(self, attrs, expected, 0.0001)
        assertTensorAlmostEqual(self, delta, expected_delta, 0.0001)

    def _deeplift_assert(
        self,
        model: Module,
        attr_method: Union[DeepLift, DeepLiftShap],
        inputs: Tuple[Tensor, ...],
        baselines,
        custom_attr_func: Callable[..., Tuple[Tensor, ...]] = None,
    ) -> None:
        input_bsz = len(inputs[0])
        if callable(baselines):
            baseline_parameters = signature(baselines).parameters
            if len(baseline_parameters) > 0:
                baselines = baselines(inputs)
            else:
                baselines = baselines()

        baseline_bsz = (
            len(baselines[0]) if isinstance(baselines[0], torch.Tensor) else 1
        )
        # Run attribution multiple times to make sure that it is
        # working as expected
        for _ in range(5):
            model.zero_grad()
            attributions, delta = attr_method.attribute(
                inputs,
                baselines,
                return_convergence_delta=True,
                custom_attribution_func=custom_attr_func,
            )
            attributions_without_delta = attr_method.attribute(
                inputs, baselines, custom_attribution_func=custom_attr_func
            )

            for attribution, attribution_without_delta in zip(
                attributions, attributions_without_delta
            ):
                self.assertTrue(
                    torch.all(torch.eq(attribution, attribution_without_delta))
                )

            if isinstance(attr_method, DeepLiftShap):
                self.assertEqual([input_bsz * baseline_bsz], list(delta.shape))
            else:
                self.assertEqual([input_bsz], list(delta.shape))
                delta_external = attr_method.compute_convergence_delta(
                    attributions, baselines, inputs
                )
                assertTensorAlmostEqual(
                    self, delta, delta_external, delta=0.0, mode="max"
                )

            delta_condition = (delta.abs() < 0.00001).all()
            self.assertTrue(
                delta_condition,
                "The sum of attribution values {} is not "
                "nearly equal to the difference between the endpoint for "
                "some samples".format(delta),
            )
            for input, attribution in zip(inputs, attributions):
                self.assertEqual(input.shape, attribution.shape)
            if (
                isinstance(baselines[0], (int, float))
                or inputs[0].shape == baselines[0].shape
            ):
                # Compare with Integrated Gradients
                ig = IntegratedGradients(model)
                attributions_ig = ig.attribute(inputs, baselines)
                assertAttributionComparision(self, attributions, attributions_ig)


def _hypothetical_contrib_func(
    multipliers: Tuple[Tensor, ...],
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    r"""
    Implements hypothetical input contributions based on the logic described here:
    https://github.com/kundajelab/deeplift/pull/36/files
    This is using a dummy model for test purposes
    """
    # we assume that multiplies, inputs and baselines have the following shape:
    # tuple((bsz x len x channel), )
    assert len(multipliers[0].shape) == 3, multipliers[0].shape
    assert len(inputs[0].shape) == 3, inputs[0].shape
    assert len(baselines[0].shape) == 3, baselines[0].shape
    assert len(multipliers) == len(inputs) and len(inputs) == len(baselines), (
        "multipliers, inputs and baselines must have the same shape but"
        "multipliers: {}, inputs: {}, baselines: {}".format(
            len(multipliers), len(inputs), len(baselines)
        )
    )

    attributions = []
    for k in range(len(multipliers)):
        sub_attributions = torch.zeros_like(inputs[k])
        for i in range(inputs[k].shape[-1]):
            hypothetical_input = torch.zeros_like(inputs[k])
            hypothetical_input[:, :, i] = 1.0
            hypothetical_input_ref_diff = hypothetical_input - baselines[k]
            sub_attributions[:, :, i] = torch.sum(
                hypothetical_input_ref_diff * multipliers[k], dim=-1
            )
        attributions.append(sub_attributions)
    return tuple(attributions)
