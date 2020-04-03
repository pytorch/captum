#!/usr/bin/env python3

import numpy as np
import torch

from captum.attr import DeepLift, IntegratedGradients, Saliency
from captum.metrics import infidelity

from ..helpers.basic import BaseTest, assertArraysAlmostEqual
from ..helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)


def _local_perturb_func(input1, input2=None):
    perturb1 = torch.stack(
        [torch.Tensor([0.0009])] * np.prod(list(input1.shape)), dim=-1
    ).view(input1.shape)
    if input2 is not None:
        perturb2 = torch.stack(
            [torch.Tensor([0.0121])] * np.prod(list(input2.shape)), dim=-1
        ).view(input2.shape)
        return (perturb1, perturb2), (input1 - perturb1, input2 - perturb2)
    return perturb1, input1 - perturb1


def _global_perturb_func1(input1, input2):
    pert1 = torch.ones(input1.shape)
    pert2 = torch.ones(input2.shape)

    return (pert1, pert2), (torch.zeros(input1.shape), torch.zeros(input2.shape))


def _global_perturb_func2(input1, input2):
    pert1 = torch.tensor(np.random.choice(2, input1.shape))
    pert2 = torch.tensor(np.random.choice(2, input2.shape))

    return (pert1, pert2), ((1 - pert1) * input1, (1 - pert2) * input2)


class Test(BaseTest):
    def test_basic_infidelity_single(self):
        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        inputs = (input1, input2)
        expected = [0.0000316]

        self.basic_model_local_assert(BasicModel2(), inputs, expected)

    def test_basic_infidelity_multiple(self):
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        expected = [0.0000316] * 3

        self.basic_model_local_assert(BasicModel2(), inputs, expected)

    def test_basic_infidelity_multiple_with_batching(self):
        input1 = torch.tensor([3.0] * 20)
        input2 = torch.tensor([1.0] * 20)
        expected = [0.0000316] * 20

        infid1 = self.basic_model_local_assert(
            BasicModel2(),
            (input1, input2),
            expected,
            n_perturb_samples=5,
            max_batch_size=21,
        )
        infid2 = self.basic_model_local_assert(
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

        self.basic_model_global_assert(
            model,
            (input1, input2),
            [0.0],
            additional_args=args,
            n_perturb_samples=4,
            max_batch_size=10,
            perturb_func=_global_perturb_func1,
        )

        self.basic_model_global_assert(
            model,
            (input1, input2),
            [0.0],
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=2,
            perturb_func=_global_perturb_func2,
        )

    def basic_model_local_assert(
        self, model, inputs, expected, n_perturb_samples=10, max_batch_size=None
    ):
        ig = IntegratedGradients(model)

        attrs = tuple(attr / input for input, attr in zip(inputs, ig.attribute(inputs)))

        return self.infidelity_assert(
            model,
            attrs,
            inputs,
            expected,
            n_perturb_samples=n_perturb_samples,
            max_batch_size=max_batch_size,
        )

    def basic_model_global_assert(
        self,
        model,
        inputs,
        expected,
        additional_args=None,
        n_perturb_samples=10,
        max_batch_size=None,
        perturb_func=_global_perturb_func2,
    ):
        ig = IntegratedGradients(model)
        attrs, delta = ig.attribute(
            inputs,
            additional_forward_args=additional_args,
            return_convergence_delta=True,
        )

        infid = self.infidelity_assert(
            model,
            attrs,
            inputs,
            expected,
            additional_args=additional_args,
            perturb_func=perturb_func,
            n_perturb_samples=n_perturb_samples,
            max_batch_size=max_batch_size,
        )
        return infid

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
        additional_forward_args = (None, True)
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

    def infidelity_assert(
        self,
        model,
        attributions,
        inputs,
        expected,
        additional_args=None,
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
            n_samples=n_perturb_samples,
            max_examples_per_batch=max_batch_size,
        )
        assertArraysAlmostEqual(infid.numpy(), expected, 0.0001)
        return infid
