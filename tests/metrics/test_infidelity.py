#!/usr/bin/env python3

import numpy as np
import torch

from captum.attr import (
    DeepLift,
    FeatureAblation,
    IntegratedGradients,
    NoiseTunnel,
    Saliency,
)
from captum.metrics import infidelity_attr

from ..helpers.basic import BaseTest, assertArraysAlmostEqual
from ..helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel5_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)


class Test(BaseTest):
    def test_basic_infidelity_single(self):
        model = BasicModel2()
        ig = IntegratedGradients(model)
        fa = FeatureAblation(model)

        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        expected = [0.0002]
        self.basic_infidelity_helper(model, ig, (input1, input2), expected)

        expected = [0.0005]
        self.basic_infidelity_helper(model, fa, (input1, input2), expected)

    def test_basic_infidelity_multiple(self):
        model = BasicModel2()
        ig = IntegratedGradients(model)

        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        expected = [0.0002] * 3
        self.basic_infidelity_helper(model, ig, (input1, input2), expected)

    def test_basic_infidelity_multiple_with_batching(self):
        model = BasicModel2()
        ig = IntegratedGradients(model)

        input1 = torch.tensor([3.0] * 20)
        input2 = torch.tensor([1.0] * 20)
        expected = [0.0002] * 20

        infid1 = self.basic_infidelity_helper(
            model, ig, (input1, input2), expected, n_perturb_samples=5, max_batch_size=5
        )
        infid2 = self.basic_infidelity_helper(
            model,
            ig,
            (input1, input2),
            expected,
            n_perturb_samples=5,
            max_batch_size=21,
        )
        assertArraysAlmostEqual(infid1, infid2, 0.0)

    def test_basic_infidelity_additional_forward_args1(self):
        model = BasicModel4_MultiArgs()
        ig = IntegratedGradients(model)

        input1 = torch.tensor([[1.5, 2.0, 3.3]] * 7)
        input2 = torch.tensor([[3.0, 3.5, 2.2]] * 7)

        args = torch.tensor([[1.0, 3.0, 4.0]] * 7)
        self.basic_infidelity_helper(
            model,
            ig,
            (input1, input2),
            [0.0] * 7,
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=6,
        )

    def test_basic_infidelity_additional_forward_args2(self):
        model = BasicModel5_MultiArgs()
        ig = IntegratedGradients(model)

        input1 = torch.tensor([[1.5, 2.0, 3.3]] * 2)
        input2 = torch.tensor([[3.0, 3.5, 2.2]] * 2)

        args = ([2, 3], 1)
        self.basic_infidelity_helper(
            model,
            ig,
            (input1, input2),
            [0.0] * 2,
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=1,
        )

    def test_classification_infidelity_convnet(self):
        model = BasicModel_ConvNet_One_Conv()
        dl = DeepLift(model)

        input = torch.stack([torch.arange(16).float()] * 2, dim=0).view(2, 1, 4, 4)

        self.basic_infidelity_helper(
            model, dl, input, [0.0032, 0.0032], target=0, multi_input=False
        )

    def test_classification_infidelity_convnet_multi_targets(self):
        model = BasicModel_ConvNet_One_Conv()
        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)

        input = torch.stack([torch.arange(16).float()] * 20, dim=0).view(20, 1, 4, 4)

        self.basic_infidelity_helper(
            model,
            ig,
            input,
            [0.0032] * 20,
            target=torch.tensor([1] * 20),
            multi_input=False,
            n_perturb_samples=500,
            max_batch_size=10,
        )
        self.basic_infidelity_helper(
            model,
            nt,
            input,
            [0.0001] * 20,
            target=torch.tensor([1] * 20),
            multi_input=False,
            n_perturb_samples=500,
            max_batch_size=10,
            nt_type="vargrad",
        )

    def test_classification_infidelity_tpl_target(self):
        model = BasicModel_MultiLayer()
        input = torch.randn(4, 3)
        additional_forward_args = (None, True)
        targets = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        sa = Saliency(model)

        infid1 = self.basic_infidelity_helper(
            model,
            sa,
            input,
            [0.0, 0.0167, 0.0167, 0.0167],
            additional_args=additional_forward_args,
            target=targets,
            multi_input=False,
        )

        infid2 = self.basic_infidelity_helper(
            model,
            sa,
            input,
            [0.0, 0.0167, 0.0167, 0.0167],
            additional_args=additional_forward_args,
            target=targets,
            max_batch_size=2,
            multi_input=False,
        )
        assertArraysAlmostEqual(infid1, infid2, 1e-05)

    def basic_infidelity_helper(
        self,
        model,
        attr_algo,
        inputs,
        expected,
        additional_args=None,
        n_perturb_samples=10,
        target=None,
        max_batch_size=None,
        multi_input=True,
        **kwargs
    ):
        def perturb_func(input1, input2=None):
            perturb1 = torch.stack(
                [torch.Tensor([0.0009])] * np.prod(list(input1.shape)), dim=-1
            ).view(input1.shape)
            if multi_input:
                perturb2 = torch.stack(
                    [torch.Tensor([0.0121])] * np.prod(list(input2.shape)), dim=-1
                ).view(input2.shape)
                return (perturb1, perturb2), (input1 + perturb1, input2 + perturb2)
            return perturb1, input1 + perturb1

        attrs = attr_algo.attribute(
            inputs, additional_forward_args=additional_args, target=target, **kwargs
        )
        infid = infidelity_attr(
            model,
            perturb_func,
            inputs,
            attrs,
            additional_forward_args=additional_args,
            target=target,
            n_samples=n_perturb_samples,
            max_examples_per_batch=max_batch_size,
        )
        assertArraysAlmostEqual(infid.numpy(), expected, 0.0001)
        return infid
