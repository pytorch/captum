import torch
from torch import Tensor
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
from captum.attr._core.neuron.neuron_gradient import NeuronGradient
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from captum.attr._core.neuron.neuron_gradient_shap import NeuronGradientShap
from captum.attr._core.neuron.neuron_feature_ablation import NeuronFeatureAblation

from .basic_models import BasicModel_MultiLayer


config = [
    # Attribution Method Configs
    {
        "name": "basic_multiple_target_with_baselines",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift, DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": [0, 1, 1, 0]},
    }, 
    {
        "name": "basic_multiple_tuple_target_with_baselines",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift, DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)], "additional_forward_args": (None, True)},
    }, 
     {
        "name": "basic_multiple_tuple_target",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_tensor_single_target",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0]),
        },
    },
    {
        "name": "basic_tensor_multi_target",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([1, 1, 0, 0]),
        },
    },
    {
        "name": "basic_multiple_tuple_target_with_internal_batching",
        "algorithms": [IntegratedGradients],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "internal_batch_size": 2,
        },
    },
    {
        "name": "basic_multiple_target_with_varying_baselines",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(2,3), "target": [0, 1, 1, 0], "n_samples": 500},
        "target_delta": 0.6,
    },
    {
        "name": "basic_multiple_tuple_target_with_ablations_per_eval",
        "algorithms": [FeatureAblation],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "ablations_per_eval": 2,
        },
    },
    {
        "name": "basic_multiple_tuple_target_with_ablations_per_eval_occlusion",
        "algorithms": [Occlusion],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "ablations_per_eval": 2,
            "sliding_window_shapes": (2,),
        },
    },
    # Layer Attribution Method Configs
    {
        "name": "basic_layer_multiple_target",
        "algorithms": [LayerConductance, LayerIntegratedGradients, LayerDeepLift, InternalInfluence, LayerFeatureAblation, LayerGradientXActivation],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0]},
    },
    {
        "name": "basic_layer_tensor_multiple_target",
        "algorithms": [LayerConductance, LayerIntegratedGradients, LayerDeepLift, InternalInfluence, LayerFeatureAblation, LayerGradientXActivation],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": torch.tensor([0, 1, 1, 0])},
    },
    {
        "name": "basic_layer_multiple_tuple_target",
        "algorithms": [LayerConductance, LayerIntegratedGradients, LayerDeepLift, InternalInfluence, LayerFeatureAblation, LayerGradientXActivation],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_layer_multiple_tuple_target_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "internal_batch_size": 2,
        },
    },
    # Neuron Attribution Method Configs
    {
        "name": "basic_neuron_multiple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0], "neuron_index": 3},
    },
    {
        "name": "basic_neuron_tensor_multiple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": torch.tensor([0, 1, 1, 0]), "neuron_index": 3},
    },
    {
        "name": "basic_neuron_multiple_tuple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "neuron_index": 3
        },
    },
    {
        "name": "basic_neuron_multiple_tuple_target_with_internal_batching",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "internal_batch_size": 2,
            "neuron_index": 3
        },
    }
]


"""

    {


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

"""
