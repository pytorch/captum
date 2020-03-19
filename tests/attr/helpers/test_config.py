import torch
import random
import numpy as np
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.saliency import Saliency
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.occlusion import Occlusion
from captum.attr._core.guided_backprop_deconvnet import GuidedBackprop, Deconvolution
from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.attr._core.shapley_value import ShapleyValueSampling
from captum.attr._core.feature_permutation import FeaturePermutation

from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap

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

from .basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModel_ConvNet,
    ReLULinearDeepLiftModel,
)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
tconfig = [
    {
        "name": "basic_multiple_tuple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.15 * torch.randn(1, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "n_samples": 2000,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    }
]
config = [
    # Attribution Method Configs
    # Primary Methods (Generic Configs)
    {
        "name": "basic_single_target",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 1},
    },
    {
        "name": "basic_multi_input",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
    },
    {
        "name": "basic_multi_target",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0]},
    },
    {
        "name": "basic_multi_input_multi_target",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
        },
    },
    {
        "name": "basic_multiple_tuple_target",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_tensor_single_target",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": torch.tensor([0])},
    },
    {
        "name": "basic_tensor_multi_target",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([1, 1, 0, 0]),
        },
    },
    # Primary Configs with Baselines
    {
        "name": "basic_multiple_tuple_target_with_baselines",
        "algorithms": [
            IntegratedGradients,
            FeatureAblation,
            DeepLift,
            ShapleyValueSampling,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_tensor_single_target_with_baselines",
        "algorithms": [
            IntegratedGradients,
            FeatureAblation,
            DeepLift,
            ShapleyValueSampling,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
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
        "name": "basic_multi_input_multi_target_nt",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "n_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_multiple_target_with_baseline_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [0, 1, 1, 0],
            "n_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_multiple_tuple_target_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "n_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_single_tensor_target_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0]),
            "n_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_multi_tensor_target_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "n_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    # DeepLift SHAP
    {
        "name": "basic_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(6, 3),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_input_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "baselines": (torch.randn(4, 3), torch.randn(4, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_target_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [0, 1, 1, 0],
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_tuple_target_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_single_tensor_targe_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": torch.tensor([0]),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_tensor_target_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
        },
        "baseline_distr": True,
    },
    # Gradient SHAP
    {
        "name": "basic_multi_inp_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (torch.randn(6, 3), torch.randn(6, 3)),
            "baselines": (torch.randn(1, 3), torch.randn(1, 3)),
            "additional_forward_args": (torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "stdevs": 0.0,
            "n_samples": 2000,
        },
        "target_delta": 1.0,
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(1, 3),
            "target": [0, 1, 1, 0],
            "n_samples": 800,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_tuple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.15 * torch.randn(1, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "n_samples": 2000,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_single_tensor_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(1, 3),
            "target": torch.tensor([0]),
            "n_samples": 500,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_tensor_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(1, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "n_samples": 500,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    # Perturbation-Specific Configs
    {
        "name": "conv_with_perturbations_per_eval",
        "algorithms": [FeatureAblation, ShapleyValueSampling, FeaturePermutation],
        "model": BasicModel_ConvNet(),
        "attribute_args": {
            "inputs": torch.arange(400).view(4, 1, 10, 10).float(),
            "target": 0,
            "perturbations_per_eval": 20,
        },
    },
    {
        "name": "basic_multiple_tuple_target_with_perturbations_per_eval",
        "algorithms": [FeatureAblation, ShapleyValueSampling, FeaturePermutation],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "perturbations_per_eval": 2,
        },
    },
    {
        "name": "conv_occlusion_with_perturbations_per_eval",
        "algorithms": [Occlusion],
        "model": BasicModel_ConvNet(),
        "attribute_args": {
            "inputs": torch.arange(400).view(4, 1, 10, 10).float(),
            "perturbations_per_eval": 8,
            "sliding_window_shapes": (1, 4, 2),
            "target": 0,
        },
    },
    {
        "name": "basic_multi_input_with_perturbations_per_eval_occlusion",
        "algorithms": [Occlusion],
        "model": ReLULinearDeepLiftModel(),
        "attribute_args": {
            "inputs": (torch.randn(4, 3), torch.randn(4, 3)),
            "perturbations_per_eval": 2,
            "sliding_window_shapes": ((2,), (1,)),
        },
    },
    {
        "name": "basic_multiple_tuple_target_occlusion",
        "algorithms": [Occlusion],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "sliding_window_shapes": (2,),
        },
    },
    # Layer Attribution Method Configs
    {
        "name": "conv_layer_single_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
            GuidedGradCam,
        ],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {"inputs": 100 * torch.randn(4, 1, 10, 10), "target": 1},
    },
    {
        "name": "basic_layer_in_place",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(inplace=True),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_multi_output",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_multi_input",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
    },
    {
        "name": "basic_layer_multiple_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0]},
    },
    {
        "name": "basic_layer_tensor_multiple_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
        },
    },
    {
        "name": "basic_layer_multiple_tuple_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
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
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
            "internal_batch_size": 2,
        },
    },
    {
        "name": "basic_layer_multi_output_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": 0,
            "internal_batch_size": 2,
        },
    },
    # Layer Perturbation
    {
        "name": "basic_layer_multi_input_with_perturbations_per_eval",
        "algorithms": [LayerFeatureAblation],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
            "perturbations_per_eval": 2,
        },
    },
    {
        "name": "basic_layer_multi_output_perturbations_per_eval",
        "algorithms": [LayerFeatureAblation],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": 0,
            "perturbations_per_eval": 2,
        },
    },
    {
        "name": "conv_layer_with_perturbations_per_eval",
        "algorithms": [LayerFeatureAblation],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "target": 1,
            "perturbations_per_eval": 20,
        },
    },
    # Layer DeepLiftSHAP
    {
        "name": "relu_layer_multi_inp_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": ReLULinearDeepLiftModel(),
        "layer": "l3",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (2 * torch.randn(2, 3), 6 * torch.randn(2, 3)),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_output_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": torch.randn(2, 3),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_inp_multi_target_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (2 * torch.randn(11, 3), 6 * torch.randn(11, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multiple_target_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(6, 3),
            "target": [0, 1, 1, 0],
        },
        "baseline_distr": True,
    },
    # Layer Gradient SHAP
    {
        "name": "relu_layer_multi_inp_grad_shap",
        "algorithms": [LayerGradientShap],
        "model": ReLULinearDeepLiftModel(),
        "layer": "l3",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (2 * torch.randn(2, 3), 6 * torch.randn(2, 3)),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_output_grad_shap",
        "algorithms": [LayerGradientShap],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": torch.randn(2, 3),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_inp_multi_target_grad_shap",
        "algorithms": [LayerGradientShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (torch.randn(6, 3), torch.randn(6, 3)),
            "baselines": (torch.randn(2, 3), torch.randn(2, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "n_samples": 1000,
        },
        "baseline_distr": True,
        "target_delta": 0.6,
    },
    # Neuron Attribution Method Configs
    {
        "name": "basic_neuron",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "neuron_index": 3},
    },
    {
        "name": "conv_neuron",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "neuron_index": (0, 1, 0),
        },
    },
    {
        "name": "basic_neuron_multi_input",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_index": (3,),
        },
    },
    # Neuron Conductance (with target)
    {
        "name": "basic_neuron_single_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 1, "neuron_index": 3},
    },
    {
        "name": "basic_neuron_multiple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [0, 1, 1, 0],
            "neuron_index": 3,
        },
    },
    {
        "name": "conv_neuron_single_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "target": 1,
            "neuron_index": (0, 1, 0),
        },
    },
    {
        "name": "basic_neuron_multi_input_multi_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "neuron_index": 3,
        },
    },
    {
        "name": "basic_neuron_tensor_multiple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "neuron_index": 3,
        },
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
            "neuron_index": 3,
        },
    },
    # Neuron Conductance with Internal Batching
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
            "neuron_index": 3,
        },
    },
    {
        "name": "basic_neuron_multi_input_multi_target_with_internal_batching",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "internal_batch_size": 2,
            "neuron_index": 3,
        },
    },
    # Neuron Gradient SHAP
    {
        "name": "basic_neuron_grad_shap",
        "algorithms": [NeuronGradientShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": torch.randn(1, 3),
            "neuron_index": 3,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_neuron_multi_inp_grad_shap",
        "algorithms": [NeuronGradientShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (10 * torch.randn(1, 3), 5 * torch.randn(1, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "neuron_index": 3,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    # Neuron DeepLift SHAP
    {
        "name": "basic_neuron_dl_shap",
        "algorithms": [NeuronDeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(6, 3),
            "neuron_index": (3,),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_neuron_multi_input_dl_shap",
        "algorithms": [NeuronDeepLiftShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "baselines": (torch.randn(4, 3), torch.randn(4, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_index": 3,
        },
        "baseline_distr": True,
    },
    # Neuron Feature Ablation
    {
        "name": "conv_neuron_with_perturbations_per_eval",
        "algorithms": [NeuronFeatureAblation],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": torch.arange(400).view(4, 1, 10, 10).float(),
            "perturbations_per_eval": 20,
            "neuron_index": (0, 1, 0),
        },
    },
    {
        "name": "basic_neuron_multiple_input_with_baselines_and_perturbations_per_eval",
        "algorithms": [NeuronFeatureAblation],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "baselines": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_index": (3,),
            "perturbations_per_eval": 2,
        },
    },
]


"""



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

"""
