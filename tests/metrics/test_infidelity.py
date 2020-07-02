#!/usr/bin/env python3

import torch

from captum.attr import DeepLift, FeatureAblation, IntegratedGradients, Saliency
from captum.metrics import infidelity, infidelity_perturb_func_decorator

from ..helpers.basic import BaseTest, assertArraysAlmostEqual, assertTensorAlmostEqual
from ..helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)


@infidelity_perturb_func_decorator
def _local_perturb_func_default(inputs):
    return _local_perturb_func(inputs)[1]


def _local_perturb_func(inputs):
    if isinstance(inputs, tuple):
        input1 = inputs[0]
        input2 = inputs[1]
    else:
        input1 = inputs
        input2 = None

    perturb1 = 0.0009 * torch.ones_like(input1)
    if input2 is None:
        return perturb1, input1 - perturb1

    perturb2 = 0.0121 * torch.ones_like(input2)
    return (perturb1, perturb2), (input1 - perturb1, input2 - perturb2)


@infidelity_perturb_func_decorator
def _global_perturb_func1_default(inputs):
    return _global_perturb_func1(inputs)[1]


# sensitivity-N, N = #input features
def _global_perturb_func1(inputs):
    if isinstance(inputs, tuple):
        input1 = inputs[0]
        input2 = inputs[1]
    else:
        input1 = inputs
        input2 = None
    pert1 = torch.ones(input1.shape)
    if input2 is None:
        return pert1, torch.zeros(input1.shape)

    pert2 = torch.ones(input2.shape)
    return (pert1, pert2), (torch.zeros(input1.shape), torch.zeros(input2.shape))


class Test(BaseTest):
    def test_basic_infidelity_single(self):
        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        inputs = (input1, input2)
        expected = [0.0]

        self.basic_model_assert(BasicModel2(), inputs, expected)

    def test_basic_infidelity_multiple(self):
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        expected = [0.0] * 3

        infid = self.basic_model_assert(BasicModel2(), inputs, expected)
        infid_w_common_func = self.basic_model_assert(
            BasicModel2(),
            inputs,
            expected,
            perturb_func=_local_perturb_func_default,
            local=False,
        )
        assertTensorAlmostEqual(self, infid, infid_w_common_func)

    def test_basic_infidelity_multiple_with_batching(self):
        input1 = torch.tensor([3.0] * 20)
        input2 = torch.tensor([1.0] * 20)
        expected = [0.0] * 20

        infid1 = self.basic_model_assert(
            BasicModel2(),
            (input1, input2),
            expected,
            n_perturb_samples=5,
            max_batch_size=21,
        )
        infid2 = self.basic_model_assert(
            BasicModel2(),
            (input1, input2),
            expected,
            n_perturb_samples=5,
            max_batch_size=60,
        )
        assertArraysAlmostEqual(infid1, infid2, 0.0)

    def test_basic_infidelity_additional_forward_args1(self):
        model = BasicModel4_MultiArgs()

        input1 = torch.tensor([[1.5, 2.0, 3.3]])
        input2 = torch.tensor([[3.0, 3.5, 2.2]])

        args = torch.tensor([[1.0, 3.0, 4.0]])
        ig = IntegratedGradients(model)

        infidelity1 = self.basic_model_global_assert(
            ig,
            model,
            (input1, input2),
            [0.0],
            additional_args=args,
            n_perturb_samples=1,
            max_batch_size=1,
            perturb_func=_global_perturb_func1,
        )

        infidelity2 = self.basic_model_global_assert(
            ig,
            model,
            (input1, input2),
            [0.0],
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=2,
            perturb_func=_global_perturb_func1,
        )

        infidelity2_w_custom_pert_func = self.basic_model_global_assert(
            ig,
            model,
            (input1, input2),
            [0.0],
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=2,
            perturb_func=_global_perturb_func1_default,
        )
        assertTensorAlmostEqual(self, infidelity1, infidelity2, 0.0)
        assertTensorAlmostEqual(self, infidelity2_w_custom_pert_func, infidelity2, 0.0)

    def test_classification_infidelity_convnet_multi_targets(self):
        model = BasicModel_ConvNet_One_Conv()
        dl = DeepLift(model)

        input = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(20, 1, 4, 4)

        self.infidelity_assert(
            model,
            dl.attribute(input, target=torch.tensor([1] * 20)) / input,
            input,
            [0.0] * 20,
            target=torch.tensor([1] * 20),
            multi_input=False,
            n_perturb_samples=500,
            max_batch_size=120,
        )

    def test_classification_infidelity_tpl_target(self):
        model = BasicModel_MultiLayer()
        input = torch.arange(1.0, 13.0).view(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        targets = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        sa = Saliency(model)

        infid1 = self.infidelity_assert(
            model,
            sa.attribute(
                input, target=targets, additional_forward_args=additional_forward_args
            ),
            input,
            [0.0, 0.0, 0.0, 0.0],
            additional_args=additional_forward_args,
            target=targets,
            multi_input=False,
        )

        infid2 = self.infidelity_assert(
            model,
            sa.attribute(
                input, target=targets, additional_forward_args=additional_forward_args
            ),
            input,
            [0.0, 0.0, 0.0, 0.0],
            additional_args=additional_forward_args,
            target=targets,
            max_batch_size=2,
            multi_input=False,
        )
        assertArraysAlmostEqual(infid1, infid2, 1e-05)

    def test_classification_infidelity_tpl_target_w_baseline(self):
        model = BasicModel_MultiLayer()
        input = torch.arange(1.0, 13.0).view(4, 3)
        baseline = torch.ones(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        targets = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        ig = IntegratedGradients(model)

        def perturbed_func2(inputs, baselines):
            return torch.ones(baselines.shape), baselines

        @infidelity_perturb_func_decorator
        def perturbed_func3(inputs, baselines):
            return baselines

        attr, delta = ig.attribute(
            input,
            target=targets,
            additional_forward_args=additional_forward_args,
            baselines=baseline,
            return_convergence_delta=True,
        )

        infid = self.infidelity_assert(
            model,
            attr,
            input,
            [0.10686, 0.0, 0.0, 0.0],
            additional_args=additional_forward_args,
            baselines=baseline,
            target=targets,
            multi_input=False,
            n_perturb_samples=3,
            perturb_func=perturbed_func3,
        )
        infid2 = self.infidelity_assert(
            model,
            attr,
            input,
            [0.10686, 0.0, 0.0, 0.0],
            additional_args=additional_forward_args,
            baselines=baseline,
            target=targets,
            multi_input=False,
            n_perturb_samples=3,
            perturb_func=perturbed_func2,
        )

        assertTensorAlmostEqual(self, infid, delta * delta)
        assertTensorAlmostEqual(self, infid, infid2)

    def test_sensitivity_n_ig(self):
        model = BasicModel_MultiLayer()
        ig = IntegratedGradients(model)
        self.basic_multilayer_sensitivity_n(ig, model)

    def test_sensitivity_n_fa(self):
        model = BasicModel_MultiLayer()
        fa = FeatureAblation(model)
        self.basic_multilayer_sensitivity_n(fa, model)

    def basic_multilayer_sensitivity_n(self, attr_algo, model):
        # sensitivity-2
        def _global_perturb_func2(input):
            pert = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1]]).float()
            return pert, (1 - pert) * input

        # sensitivity-1
        def _global_perturb_func3(input):
            pert = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
            return pert, (1 - pert) * input

        @infidelity_perturb_func_decorator
        def _global_perturb_func3_custom(input):
            return _global_perturb_func3(input)[1]

        input = torch.tensor([[1.0, 2.5, 3.3]])

        # infidelity for sensitivity-1
        infid = self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            [0.0],
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func3,
        )

        infid_w_default = self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            [0.0],
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func3_custom,
        )
        assertTensorAlmostEqual(self, infid, infid_w_default)

        # infidelity for sensitivity-2
        self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            [0.0],
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func2,
        )

        # infidelity for sensitivity-3
        self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            [0.0],
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func1,
        )

    def basic_model_assert(
        self,
        model,
        inputs,
        expected,
        n_perturb_samples=10,
        max_batch_size=None,
        perturb_func=_local_perturb_func,
        local=True,
    ):
        ig = IntegratedGradients(model)
        if local:
            attrs = tuple(
                attr / input for input, attr in zip(inputs, ig.attribute(inputs))
            )
        else:
            attrs = ig.attribute(inputs)
        return self.infidelity_assert(
            model,
            attrs,
            inputs,
            expected,
            n_perturb_samples=n_perturb_samples,
            max_batch_size=max_batch_size,
            perturb_func=perturb_func,
        )

    def basic_model_global_assert(
        self,
        attr_algo,
        model,
        inputs,
        expected,
        additional_args=None,
        target=None,
        n_perturb_samples=10,
        max_batch_size=None,
        perturb_func=_global_perturb_func1,
    ):
        attrs = attr_algo.attribute(
            inputs, additional_forward_args=additional_args, target=target
        )
        infid = self.infidelity_assert(
            model,
            attrs,
            inputs,
            expected,
            additional_args=additional_args,
            perturb_func=perturb_func,
            target=target,
            n_perturb_samples=n_perturb_samples,
            max_batch_size=max_batch_size,
        )
        return infid

    def infidelity_assert(
        self,
        model,
        attributions,
        inputs,
        expected,
        additional_args=None,
        baselines=None,
        n_perturb_samples=10,
        target=None,
        max_batch_size=None,
        multi_input=True,
        perturb_func=_local_perturb_func,
        **kwargs
    ):
        infid = infidelity(
            model,
            perturb_func,
            inputs,
            attributions,
            additional_forward_args=additional_args,
            target=target,
            baselines=baselines,
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=max_batch_size,
        )
        assertArraysAlmostEqual(infid.numpy(), expected, 0.0001)
        return infid
