#!/usr/bin/env python3

import unittest

import torch
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.saliency import Saliency
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.occlusion import Occlusion

from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap

from captum.attr._core.neuron.neuron_conductance import NeuronConductance

from .helpers.basic_models import BasicModel_MultiLayer
from .helpers.utils import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_simple_target_missing_error(self):
        net = BasicModel_MultiLayer()
        inp = torch.zeros((1, 3))
        with self.assertRaises(AssertionError):
            attr = IntegratedGradients(net)
            attr.attribute(inp)

    def test_multi_target_error(self):
        net = BasicModel_MultiLayer()
        inp = torch.zeros((1, 3))
        with self.assertRaises(AssertionError):
            attr = IntegratedGradients(net)
            attr.attribute(inp, additional_forward_args=(None, True), target=(1, 0))

    def test_simple_target_ig(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            IntegratedGradients,
            net,
            inputs=inp,
            targets=[0, 1, 1, 0],
            test_batches=True,
        )

    def test_simple_target_ig_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            IntegratedGradients,
            net,
            inputs=inp,
            targets=torch.tensor([0, 1, 1, 0]),
            test_batches=True,
        )

    def test_simple_target_ig_single_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            IntegratedGradients,
            net,
            inputs=inp,
            targets=torch.tensor([0]),
            test_batches=True,
            splice_targets=False,
        )

    def test_multi_target_ig(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            IntegratedGradients,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            test_batches=True,
        )

    def test_simple_target_saliency(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(Saliency, net, inputs=inp, targets=[0, 1, 1, 0])

    def test_simple_target_saliency_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            Saliency, net, inputs=inp, targets=torch.tensor([0, 1, 1, 0])
        )

    def test_multi_target_saliency(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            Saliency,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_simple_target_ablation(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            FeatureAblation, net, inputs=inp, targets=[0, 1, 1, 0]
        )

    def test_simple_target_ablation_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            FeatureAblation, net, inputs=inp, targets=torch.tensor([0, 1, 1, 0])
        )

    def test_multi_target_ablation(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            FeatureAblation,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_multi_target_occlusion(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            Occlusion,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            sliding_window_shapes=(2,),
        )

    def test_simple_target_deep_lift(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(DeepLift, net, inputs=inp, targets=[0, 1, 1, 0])

    def test_multi_target_deep_lift(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            DeepLift,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_simple_target_deep_lift_shap(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            DeepLiftShap, net, inputs=inp, baselines=0.5 * inp, targets=[0, 1, 1, 0]
        )

    def test_simple_target_deep_lift_shap_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            DeepLiftShap,
            net,
            inputs=inp,
            baselines=0.5 * inp,
            targets=torch.tensor([0, 1, 1, 0]),
        )

    def test_simple_target_deep_lift_shap_single_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            DeepLiftShap,
            net,
            inputs=inp,
            baselines=0.5 * inp,
            targets=torch.tensor([0]),
            splice_targets=False,
        )

    def test_multi_target_deep_lift_shap(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            DeepLiftShap,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            baselines=0.5 * inp,
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_simple_target_gradient_shap(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            GradientShap,
            net,
            inputs=inp,
            baselines=0.5 * inp[0:1],
            n_samples=500,
            stdevs=0.0,
            targets=[0, 1, 1, 0],
            delta=0.02,
        )

    def test_simple_target_gradient_shap_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            GradientShap,
            net,
            inputs=inp,
            baselines=0.5 * inp[0:1],
            n_samples=500,
            stdevs=0.0,
            targets=torch.tensor([0, 1, 1, 0]),
            delta=0.02,
        )

    def test_simple_target_gradient_shap_single_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            GradientShap,
            net,
            inputs=inp,
            baselines=0.5 * inp[0:1],
            n_samples=500,
            stdevs=0.0,
            targets=torch.tensor([0]),
            splice_targets=False,
            delta=0.02,
        )

    def test_multi_target_gradient_shap(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            GradientShap,
            net,
            inputs=inp,
            baselines=0.5 * inp[0:1],
            n_samples=500,
            stdevs=0.0,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            delta=0.02,
        )

    def test_simple_target_nt(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NoiseTunnel,
            IntegratedGradients(net),
            inputs=inp,
            targets=[0, 1, 1, 0],
            stdevs=0.0,
            test_batches=True,
        )

    def test_simple_target_nt_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NoiseTunnel,
            IntegratedGradients(net),
            inputs=inp,
            targets=torch.tensor([0, 1, 1, 0]),
            stdevs=0.0,
            test_batches=True,
        )

    def test_simple_target_nt_single_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NoiseTunnel,
            IntegratedGradients(net),
            inputs=inp,
            targets=torch.tensor([0]),
            stdevs=0.0,
            test_batches=True,
            splice_targets=False,
        )

    def test_multi_target_nt(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NoiseTunnel,
            IntegratedGradients(net),
            inputs=inp,
            additional_forward_args=(None, True),
            stdevs=0.0,
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            test_batches=True,
        )

    def test_simple_target_input_x_gradient(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            InputXGradient, net, inputs=inp, targets=[0, 1, 1, 0]
        )

    def test_multi_target_input_x_gradient(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            InputXGradient,
            net,
            inputs=inp,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_simple_target_int_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            InternalInfluence,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=[0, 1, 1, 0],
            test_batches=True,
        )

    def test_multi_target_int_inf(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            InternalInfluence,
            net,
            inputs=inp,
            target_layer=net.relu,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            test_batches=True,
        )

    def test_simple_target_layer_cond(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerConductance,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=[0, 1, 1, 0],
            test_batches=True,
        )

    def test_simple_target_layer_cond_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerConductance,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=torch.tensor([0, 1, 1, 0]),
            test_batches=True,
        )

    def test_multi_target_layer_cond(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerConductance,
            net,
            inputs=inp,
            target_layer=net.relu,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            test_batches=True,
        )

    def test_simple_target_layer_deeplift(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerDeepLift, net, inputs=inp, target_layer=net.relu, targets=[0, 1, 1, 0],
        )

    def test_simple_target_layer_deeplift_shap(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        baseline = torch.randn(6, 3)
        self._target_batch_test_assert(
            LayerDeepLiftShap,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=[0, 1, 1, 0],
            baselines=baseline,
        )

    def test_simple_target_layer_ig(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerIntegratedGradients,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=[0, 1, 1, 0],
        )

    def test_multi_target_layer_ig(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerIntegratedGradients,
            net,
            inputs=inp,
            target_layer=net.relu,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_simple_target_layer_gradient_x_act(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerGradientXActivation,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=[0, 1, 1, 0],
        )

    def test_multi_target_layer_gradient_x_act(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerGradientXActivation,
            net,
            inputs=inp,
            target_layer=net.relu,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
        )

    def test_simple_target_layer_ablation_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            LayerFeatureAblation,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=torch.tensor([0, 1, 1, 0]),
        )

    def test_simple_target_neuron_conductance(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NeuronConductance,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=[0, 1, 1, 0],
            neuron_index=3,
            test_batches=True,
        )

    def test_simple_target_neuron_conductance_tensor(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NeuronConductance,
            net,
            inputs=inp,
            target_layer=net.relu,
            targets=torch.tensor([0, 1, 1, 0]),
            neuron_index=3,
            test_batches=True,
        )

    def test_multi_target_neuron_conductance(self):
        net = BasicModel_MultiLayer()
        inp = torch.randn(4, 3)
        self._target_batch_test_assert(
            NeuronConductance,
            net,
            inputs=inp,
            target_layer=net.relu,
            additional_forward_args=(None, True),
            targets=[(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            neuron_index=3,
            test_batches=True,
        )

    def _target_batch_test_assert(
        self,
        algorithm,
        model,
        inputs,
        targets,
        target_layer=None,
        test_batches=False,
        splice_targets=True,
        delta=0.0001,
        **kwargs
    ):
        if target_layer:
            attr_method = algorithm(model, target_layer)
        else:
            attr_method = algorithm(model)

        batch_sizes = [None]
        if test_batches:
            batch_sizes = [None, 2, 4]
        for batch_size in batch_sizes:
            if batch_size:
                attributions_orig = attr_method.attribute(
                    inputs=inputs,
                    target=targets,
                    internal_batch_size=batch_size,
                    **kwargs
                )
            else:
                attributions_orig = attr_method.attribute(
                    inputs=inputs, target=targets, **kwargs
                )
            for i in range(len(inputs)):
                single_attr = attr_method.attribute(
                    inputs=inputs[i : i + 1],
                    target=targets[i] if splice_targets else targets,
                    **kwargs
                )
                single_attr_target_list = attr_method.attribute(
                    inputs=inputs[i : i + 1],
                    target=targets[i : i + 1] if splice_targets else targets,
                    **kwargs
                )
                assertTensorAlmostEqual(
                    self, attributions_orig[i : i + 1], single_attr, delta=delta
                )
                assertTensorAlmostEqual(
                    self,
                    attributions_orig[i : i + 1],
                    single_attr_target_list,
                    delta=delta,
                )


if __name__ == "__main__":
    unittest.main()
