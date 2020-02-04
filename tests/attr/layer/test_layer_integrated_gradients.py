#!/usr/bin/env python3

from typing import List, Tuple, Union, Any, cast

import torch
from torch import Tensor
from torch.nn import Module

from ..helpers.utils import assertArraysAlmostEqual, assertTensorTuplesAlmostEqual

from captum.attr._models.base import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_conductance import LayerConductance

from ..helpers.utils import BaseTest
from ..helpers.basic_models import (
    BasicEmbeddingModel,
    BasicModel_MultiLayer,
)


class Test(BaseTest):
    def test_compare_with_emb_patching(self) -> None:
        input1 = torch.tensor([[2, 5, 0, 1]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1]])
        input3 = torch.tensor([[2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1, baseline1, additional_args=(input2, input3)
        )

    def test_compare_with_emb_patching_batch(self) -> None:
        input1 = torch.tensor([[2, 5, 0, 1], [3, 1, 1, 0]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1], [2, 3, 5, 7]])
        input3 = torch.tensor([[3, 5, 6, 7], [2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1, baseline1, additional_args=(input2, input3)
        )

    def test_compare_with_layer_conductance_attr_to_outputs(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[50.0, 50.0, 50.0]], requires_grad=True)
        self._assert_compare_with_layer_conductance(model, input)

    def test_compare_with_layer_conductance_attr_to_inputs(self) -> None:
        # Note that Layer Conductance and Layer Integrated Gradients (IG) aren't
        # exactly the same. Layer IG computes partial derivative of the output
        # with respect to the layer and sums along the straight line. While Layer
        # Conductance also computes the same partial derivatives it doesn't use
        # the straight line but a path defined by F(i) - F(i - 1).
        # However, in some cases when that path becomes close to a straight line,
        # Layer IG and Layer Conductance become numerically very close.
        model = BasicModel_MultiLayer()
        input = torch.tensor([[50.0, 50.0, 50.0]], requires_grad=True)
        self._assert_compare_with_layer_conductance(model, input, True)

    def test_multiple_tensors_compare_with_expected(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._assert_compare_with_expected(
            net,
            net.relu,
            inp,
            ([[90.0, 100.0, 100.0, 100.0]], [[90.0, 100.0, 100.0, 100.0]]),
        )

    def _assert_compare_with_layer_conductance(
        self, model: Module, input: Tensor, attribute_to_layer_input: bool = False
    ):
        lc = LayerConductance(model, model.linear2)
        # For large number of steps layer conductance and layer integrated gradients
        # become very close
        attribution, delta = lc.attribute(
            input,
            target=0,
            n_steps=1500,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        lig = LayerIntegratedGradients(model, cast(Module, model.linear2))
        attributions2, delta2 = lig.attribute(
            input,
            target=0,
            n_steps=1500,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        assertArraysAlmostEqual(attribution, attributions2, 0.01)
        assertArraysAlmostEqual(delta, delta2, 0.05)

    def _assert_compare_with_emb_patching(
        self, input: Tensor, baseline: Tensor, additional_args: Tuple[Tensor, ...]
    ):
        model = BasicEmbeddingModel(nested_second_embedding=True)
        lig = LayerIntegratedGradients(model, model.embedding1)

        attributions, delta = lig.attribute(
            input,
            baselines=baseline,
            additional_forward_args=additional_args,
            return_convergence_delta=True,
        )

        # now let's interpret with standard integrated gradients and
        # the embeddings for monkey patching
        interpretable_embedding = configure_interpretable_embedding_layer(
            model, "embedding1"
        )
        input_emb = interpretable_embedding.indices_to_embeddings(input)
        baseline_emb = interpretable_embedding.indices_to_embeddings(baseline)
        ig = IntegratedGradients(model)
        attributions_with_ig, delta_with_ig = ig.attribute(
            input_emb,
            baselines=baseline_emb,
            additional_forward_args=additional_args,
            target=0,
            return_convergence_delta=True,
        )
        remove_interpretable_embedding_layer(model, interpretable_embedding)

        assertArraysAlmostEqual(attributions, attributions_with_ig)
        assertArraysAlmostEqual(delta, delta_with_ig)

    def _assert_compare_with_expected(
        self,
        model: Module,
        target_layer: Module,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_ig: Tuple[List[List[float]], ...],
        additional_input: Any = None,
    ):
        layer_ig = LayerIntegratedGradients(model, target_layer)
        attributions = layer_ig.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        assertTensorTuplesAlmostEqual(self, attributions, expected_ig, delta=0.01)
