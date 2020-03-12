import torch
from torch import Tensor
import random
import numpy as np
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

from .basic_models import BasicModel_MultiLayer, BasicModel_MultiLayer_MultiInput

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True

config = [
    # Attribution Method Configs
    # Primary Methods (Generic Configs)
    {
        "name": "basic_multiple_target",
        "algorithms": [IntegratedGradients, InputXGradient, FeatureAblation, DeepLift, Saliency],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0]},
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
    # Primary Configs with Baselines
    {
        "name": "basic_multiple_tuple_target_with_baselines",
        "algorithms": [IntegratedGradients, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)], "additional_forward_args": (None, True)},
    }, 
    {
        "name": "basic_tensor_single_target_with_baselines",
        "algorithms": [IntegratedGradients, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5*torch.randn(4,3),
            "target": torch.tensor([0]),
        },
    },
    # Primary Configs with Internal Batching
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
    # NoiseTunnel
    {
        "name": "basic_multiple_target_with_baseline_nt",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0], "n_samples": 20, "stdevs": 0.0,},
        "noise_tunnel": True,
    },
    {
        "name": "basic_multiple_tuple_target_nt",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)], "additional_forward_args": (None, True), "n_samples": 20, "stdevs": 0.0,},
        "noise_tunnel": True,
    },
    {
        "name": "basic_single_tensor_target_nt",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": torch.tensor([0]), "n_samples": 20, "stdevs": 0.0,},
        "noise_tunnel": True,
    },
    {
        "name": "basic_multi_tensor_target_nt",
        "algorithms": [IntegratedGradients, Saliency, InputXGradient, FeatureAblation, DeepLift],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": torch.tensor([0, 1, 1, 0]), "n_samples": 20, "stdevs": 0.0,},
        "noise_tunnel": True,
    },
    # DeepLift SHAP
    {
        "name": "basic_multiple_target_with_single_baseline_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": [0, 1, 1, 0]},
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_tuple_target_with_single_baseline_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)], "additional_forward_args": (None, True)},
        "baseline_distr": True,
    },
    {
        "name": "basic_single_tensor_target_with_single_baseline_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": torch.tensor([0]),},
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_tensor_target_with_single_baseline_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(4,3), "target": torch.tensor([0, 1, 1, 0])},
        "baseline_distr": True,
    },
    # Gradient SHAP
    {
        "name": "basic_multiple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(1,3), "target": [0, 1, 1, 0], "n_samples": 800, "stdevs": 0.0,},
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_tuple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(1,3), "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)], "additional_forward_args": (None, True), "n_samples": 500, "stdevs": 0.0,},
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_single_tensor_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(1,3), "target": torch.tensor([0]), "n_samples": 500, "stdevs": 0.0,},
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_tensor_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(1,3), "target": torch.tensor([0, 1, 1, 0]), "n_samples": 500, "stdevs": 0.0,},
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    # Perturbation-Specific Configs
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
        "name": "basic_layer_in_place",
        "algorithms": [LayerConductance, LayerIntegratedGradients, LayerDeepLift, InternalInfluence, LayerFeatureAblation, LayerGradientXActivation],
        "model": BasicModel_MultiLayer(inplace=True),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_multi_output",
        "algorithms": [LayerConductance, LayerIntegratedGradients, LayerDeepLift, InternalInfluence, LayerFeatureAblation, LayerGradientXActivation],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_multi_input",
        "algorithms": [LayerConductance, LayerIntegratedGradients, LayerDeepLift, InternalInfluence, LayerFeatureAblation, LayerGradientXActivation],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {"inputs": (10*torch.randn(12, 3), 5 * torch.randn(12, 3)) , "additional_forward_args": (2*torch.randn(12, 3), 5), "target": 0},
    },
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
    {
        "name": "basic_layer_multi_input_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {"inputs": (10*torch.randn(12, 3), 5 * torch.randn(12, 3)) , "additional_forward_args": (2*torch.randn(12, 3), 5), "target": 0, "internal_batch_size": 2},
    },
    {
        "name": "basic_layer_multi_output_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0, "internal_batch_size": 2},
    },
    {
        "name": "basic_multiple_target_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "baselines": 0.5*torch.randn(6,3), "target": [0, 1, 1, 0]},
        "baseline_distr": True,
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
    },
]


"""



  
    def test_multi_input_internal_inf(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            InternalInfluence,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=0,
            test_batches=True,
        )



    def test_multi_dim_layer_conductance(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerConductance, net, net.conv2, alt_device_ids=True, inputs=inp, target=1
        )

  

    def test_multi_dim_layer_integrated_gradients(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerIntegratedGradients,
            net,
            net.conv2,
            alt_device_ids=True,
            inputs=inp,
            target=1,
        )

    
  
    def test_multi_dim_layer_grad_cam(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerGradCam, net, net.conv2, alt_device_ids=True, inputs=inp, target=1
        )

    def test_multi_output_layer_grad_cam(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerGradCam, net, net.relu, alt_device_ids=True, inputs=inp, target=1
        )

  

    def test_multi_dim_layer_ablation(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        for ablations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                LayerFeatureAblation,
                net,
                net.conv2,
                alt_device_ids=True,
                inputs=inp,
                target=1,
                ablations_per_eval=ablations_per_eval,
            )

    def test_simple_neuron_conductance(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            NeuronConductance, net, net.relu, inputs=inp, neuron_index=3, target=1
        )

    def test_multi_input_neuron_conductance(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronConductance,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
            neuron_index=(3,),
            test_batches=True,
        )

    def test_multi_dim_neuron_conductance(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            NeuronConductance,
            net,
            net.conv2,
            alt_device_ids=True,
            inputs=inp,
            target=1,
            neuron_index=(0, 1, 0),
        )

    def test_simple_neuron_gradient(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            NeuronGradient,
            net,
            net.relu,
            alt_device_ids=True,
            inputs=inp,
            neuron_index=3,
        )

    def test_multi_input_neuron_gradient(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronGradient,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
        )

    def test_simple_neuron_integrated_gradient(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            NeuronIntegratedGradients,
            net,
            net.relu,
            alt_device_ids=True,
            inputs=inp,
            neuron_index=3,
        )

    def test_multi_input_integrated_gradient(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronIntegratedGradients,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
            test_batches=True,
        )

    def test_multi_input_guided_backprop(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronGuidedBackprop,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
        )

    def test_multi_input_deconv(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronDeconvolution,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
        )

    def test_multi_input_deeplift(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor(
            [[-10.0, 1.0, -5.0], [1.9, 2.0, 1.9]], requires_grad=True
        ).cuda()
        inp2 = torch.tensor(
            [[3.0, 3.0, 1.0], [1.2, 3.0, 2.3]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            DeepLift,
            net,
            None,
            inputs=(inp1, inp2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_layer_deeplift(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor(
            [[-10.0, 1.0, -5.0], [1.0, 2.0, 3.0]], requires_grad=True
        ).cuda()
        inp2 = torch.tensor(
            [[3.0, 3.0, 1.0], [4.5, 6.3, 2.3]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            LayerDeepLift,
            net,
            net.l3,
            inputs=(inp1, inp2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_layer_deeplift_shap(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()
        base2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            LayerDeepLiftShap,
            net,
            net.l3,
            inputs=(inp1, inp2),
            baselines=(base1, base2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_output_layer_deeplift_shap(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()
        base2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            LayerDeepLiftShap,
            net,
            net.relu,
            inputs=(inp1, inp2),
            target=0,
            baselines=(base1, base2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_neuron_deeplift(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        self._data_parallel_test_assert(
            NeuronDeepLift,
            net,
            net.l3,
            inputs=(inp1, inp2),
            neuron_index=0,
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_neuron_deeplift_shap(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()
        base2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            NeuronDeepLiftShap,
            net,
            net.l3,
            inputs=(inp1, inp2),
            neuron_index=0,
            baselines=(base1, base2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_basic_gradient_shap_helper(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, GradientShap, None)

    def test_basic_gradient_shap_helper_with_alt_devices(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, GradientShap, None, True)

    def test_basic_neuron_gradient_shap(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, NeuronGradientShap, net.linear2, False)

    def test_basic_neuron_gradient_shap_with_alt_devices(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, NeuronGradientShap, net.linear2, True)

    def test_basic_layer_gradient_shap(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(
            net, LayerGradientShap, net.linear1,
        )

    def test_basic_layer_gradient_shap_with_alt_devices(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(
            net, LayerGradientShap, net.linear1, True,
        )

    def _basic_gradient_shap_helper(
        self, net, attr_method_class, layer, alt_device_ids=False
    ):
        net.eval()
        inputs = torch.tensor([[1.0, -20.0, 10.0], [11.0, 10.0, -11.0]]).cuda()
        baselines = torch.randn(30, 3).cuda()
        if attr_method_class == NeuronGradientShap:
            self._data_parallel_test_assert(
                attr_method_class,
                net,
                layer,
                alt_device_ids=alt_device_ids,
                inputs=inputs,
                neuron_index=0,
                baselines=baselines,
                additional_forward_args=None,
                test_batches=False,
            )
        else:
            self._data_parallel_test_assert(
                attr_method_class,
                net,
                layer,
                alt_device_ids=alt_device_ids,
                inputs=inputs,
                target=0,
                baselines=baselines,
                additional_forward_args=None,
                test_batches=False,
            )

    def test_multi_input_neuron_ablation(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()
        for ablations_per_eval in [1, 2, 3]:
            self._data_parallel_test_assert(
                NeuronFeatureAblation,
                net,
                net.l3,
                inputs=(inp1, inp2),
                neuron_index=0,
                additional_forward_args=None,
                test_batches=False,
                ablations_per_eval=ablations_per_eval,
                alt_device_ids=True,
            )

    def test_multi_input_neuron_ablation_with_baseline(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor([[1.0, 0.0, 1.0]], requires_grad=True).cuda()
        base2 = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True).cuda()
        for ablations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                NeuronFeatureAblation,
                net,
                net.l3,
                inputs=(inp1, inp2),
                neuron_index=0,
                baselines=(base1, base2),
                additional_forward_args=None,
                test_batches=False,
                ablations_per_eval=ablations_per_eval,
            )

    def test_simple_feature_ablation(self):
        net = BasicModel_ConvNet().cuda()
        inp = torch.arange(400).view(4, 1, 10, 10).type(torch.FloatTensor).cuda()
        for ablations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                FeatureAblation,
                net,
                None,
                inputs=inp,
                target=0,
                ablations_per_eval=ablations_per_eval,
            )

    def test_simple_occlusion(self):
        net = BasicModel_ConvNet().cuda()
        inp = torch.arange(400).view(4, 1, 10, 10).float().cuda()
        for ablations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                Occlusion,
                net,
                None,
                inputs=inp,
                sliding_window_shapes=(1, 4, 2),
                target=0,
                ablations_per_eval=ablations_per_eval,
            )

    def test_multi_input_occlusion(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]]).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]]).cuda()
        for ablations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                Occlusion,
                net,
                None,
                inputs=(inp1, inp2),
                sliding_window_shapes=((2,), (1,)),
                test_batches=False,
                ablations_per_eval=ablations_per_eval,
            )

    def _alt_device_list(self):
        return [0] + [x for x in range(torch.cuda.device_count() - 1, 0, -1)]

    def _data_parallel_test_assert(
        self,
        algorithm,
        model,
        target_layer=None,
        alt_device_ids=False,
        test_batches=False,
        **kwargs
    ):
        if alt_device_ids:
            dp_model = torch.nn.parallel.DataParallel(
                model, device_ids=self._alt_device_list()
            )
        else:
            dp_model = torch.nn.parallel.DataParallel(model)
        if target_layer:
            attr_orig = algorithm(model, target_layer)
            if alt_device_ids:
                attr_dp = algorithm(
                    dp_model.forward, target_layer, device_ids=self._alt_device_list()
                )
            else:
                attr_dp = algorithm(dp_model, target_layer)
        else:
            attr_orig = algorithm(model)
            attr_dp = algorithm(dp_model)

        batch_sizes = [None]
        delta_orig = None
        delta_dp = None
        if test_batches:
            batch_sizes = [None, 1, 8]
        for batch_size in batch_sizes:
            if batch_size:
                attributions_orig = attr_orig.attribute(
                    internal_batch_size=batch_size, **kwargs
                )
            else:
                if attr_orig.has_convergence_delta():
                    attributions_orig, delta_orig = attr_orig.attribute(
                        return_convergence_delta=True, **kwargs
                    )
                else:
                    attributions_orig = attr_orig.attribute(**kwargs)
            self.setUp()
            if batch_size:
                attributions_dp = attr_dp.attribute(
                    internal_batch_size=batch_size, **kwargs
                )
            else:
                if attr_orig.has_convergence_delta():
                    attributions_dp, delta_dp = attr_dp.attribute(
                        return_convergence_delta=True, **kwargs
                    )
                else:
                    attributions_dp = attr_dp.attribute(**kwargs)

            if isinstance(attributions_dp, torch.Tensor):
                self.assertAlmostEqual(
                    torch.sum(torch.abs(attributions_orig - attributions_dp)),
                    0,
                    delta=0.0001,
                )
            else:
                for i in range(len(attributions_orig)):
                    self.assertAlmostEqual(
                        torch.sum(torch.abs(attributions_orig[i] - attributions_dp[i])),
                        0,
                        delta=0.0001,
                    )

            if delta_dp is not None:
                self.assertAlmostEqual(
                    torch.sum(torch.abs(delta_orig - delta_dp)), 0, delta=0.0001
                )


"""
