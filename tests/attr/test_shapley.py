#!/usr/bin/env python3

import io
import unittest
import unittest.mock
from typing import Any, Callable, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from tests.helpers.basic import BaseTest, assertTensorTuplesAlmostEqual
from tests.helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModelBoolInput,
)


class Test(BaseTest):
    def test_simple_shapley_sampling(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._shapley_test_assert(
            net,
            inp,
            [[76.66666, 196.66666, 116.66666]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=250,
        )

    def test_simple_shapley_sampling_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._shapley_test_assert(
            net,
            inp,
            [[275.0, 275.0, 115.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_shapley_sampling_boolean(self) -> None:
        net = BasicModelBoolInput()
        inp = torch.tensor([[True, False, True]])
        self._shapley_test_assert(
            net,
            inp,
            [[35.0, 35.0, 35.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_shapley_sampling_boolean_with_baseline(self) -> None:
        net = BasicModelBoolInput()
        inp = torch.tensor([[True, False, True]])
        self._shapley_test_assert(
            net,
            inp,
            [[-40.0, -40.0, 0.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=True,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_shapley_sampling_with_baselines(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]])
        self._shapley_test_assert(
            net,
            inp,
            [[248.0, 248.0, 104.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_shapley_sampling(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]])
        self._shapley_test_assert(
            net,
            inp,
            [[7.0, 32.5, 10.5], [76.66666, 196.66666, 116.66666]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=200,
        )

    def test_multi_sample_shapley_sampling_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._shapley_test_assert(
            net,
            inp,
            [[39.5, 39.5, 10.5], [275.0, 275.0, 115.0]],
            feature_mask=mask,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_shapley_sampling_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]])
        expected = (
            [[90, 0, 0], [78.0, 198.0, 118.0]],
            [[78, 0, 198], [0.0, 398.0, 0.0]],
            [[0, 398, 38], [0.0, 38.0, 0.0]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=200,
            test_true_shapley=False,
        )

    def test_multi_input_shapley_sampling_with_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected = (
            [[1088.6666, 1088.6666, 1088.6666], [255.0, 595.0, 255.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 595.0, 0.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 255.0, 255.0]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline = (
            [[1040, 1040, 1040], [184, 580.0, 184]],
            [[52, 1040, 132], [184, 580.0, -12.0]],
            [[52, 1040, 132], [184, 184, 184]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    # Remaining tests are for cases where forward function returns a scalar
    # per batch, as either a float, integer, 0d tensor or 1d tensor.
    def test_single_shapley_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp)).item()
        )

    def test_single_shapley_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp))
        )

    def test_single_shapley_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_single_shapley_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def test_single_shapley_int_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp.float())).item()
        )

    def test_single_shapley_int_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp.float()))
        )

    def test_single_shapley_int_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp.float())).reshape(1)
        )

    def test_single_shapley_int_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: int(torch.sum(net(inp.float())).item())
        )

    def test_multi_sample_shapley_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp)).item()
        )

    def test_multi_sample_shapley_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp))
        )

    def test_multi_sample_shapley_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_multi_sample_shapley_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def test_multi_inp_shapley_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: torch.sum(net(*inp)).item()
        )

    def test_multi_inp_shapley_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_shapley_assert(lambda *inp: torch.sum(net(*inp)))

    def test_multi_inp_shapley_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: torch.sum(net(*inp)).reshape(1)
        )

    def test_mutli_inp_shapley_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: int(torch.sum(net(*inp)).item())
        )

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_shapley_sampling_with_show_progress(self, mock_stderr) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._shapley_test_assert(
                net,
                inp,
                [[76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(bsz,),
                n_samples=250,
                show_progress=True,
            )
            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Shapley Value Sampling attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"
            assert (
                "Shapley Values attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_shapley_sampling_with_mask_and_show_progress(self, mock_stderr) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._shapley_test_assert(
                net,
                inp,
                [[275.0, 275.0, 115.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(bsz,),
                show_progress=True,
            )
            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Shapley Value Sampling attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"
            assert (
                "Shapley Values attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    def _single_input_one_sample_batch_scalar_shapley_assert(
        self, func: Callable
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._shapley_test_assert(
            func,
            inp,
            [[79.0, 79.0, 21.0]],
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def _single_input_multi_sample_batch_scalar_shapley_assert(
        self, func: Callable
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._shapley_test_assert(
            func,
            inp,
            [[629.0, 629.0, 251.0]],
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
            n_samples=2500,
        )

    def _single_int_input_multi_sample_batch_scalar_shapley_assert(
        self, func: Callable
    ) -> None:
        inp = torch.tensor([[2, 10, 3], [20, 50, 30]])
        mask = torch.tensor([[0, 0, 1]])

        self._shapley_test_assert(
            func,
            inp,
            [[629.0, 629.0, 251.0]],
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
            n_samples=2500,
        )

    def _multi_input_batch_scalar_shapley_assert(self, func: Callable) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [20.0, 10.0, 13.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2]])
        expected = (
            [[3850.6666, 3850.6666, 3850.6666]],
            [[306.6666, 3850.6666, 410.6666]],
            [[306.6666, 3850.6666, 410.6666]],
        )

        self._shapley_test_assert(
            func,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            perturbations_per_eval=(1,),
            target=None,
            n_samples=3500,
            delta=1.2,
        )

    def _shapley_test_assert(
        self,
        model: Callable,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_attr,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: Union[None, int] = 0,
        n_samples: int = 100,
        delta: float = 1.0,
        test_true_shapley: bool = True,
        show_progress: bool = False,
    ) -> None:
        for batch_size in perturbations_per_eval:
            shapley_samp = ShapleyValueSampling(model)
            attributions = shapley_samp.attribute(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
                n_samples=n_samples,
                show_progress=show_progress,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_attr, delta=delta, mode="max"
            )
            if test_true_shapley:
                shapley_val = ShapleyValues(model)
                attributions = shapley_val.attribute(
                    test_input,
                    target=target,
                    feature_mask=feature_mask,
                    additional_forward_args=additional_input,
                    baselines=baselines,
                    perturbations_per_eval=batch_size,
                    show_progress=show_progress,
                )
                assertTensorTuplesAlmostEqual(
                    self, attributions, expected_attr, mode="max", delta=0.001
                )


if __name__ == "__main__":
    unittest.main()
