from __future__ import print_function

import unittest

import torch
from captum.attr._core.grad_cam import LayerGradCam
from captum.attr._core.internal_influence import InternalInfluence
from captum.attr._core.layer_activation import LayerActivation
from captum.attr._core.layer_conductance import LayerConductance
from captum.attr._core.layer_gradient_x_activation import LayerGradientXActivation

from captum.attr._core.neuron_conductance import NeuronConductance
from captum.attr._core.neuron_gradient import NeuronGradient
from captum.attr._core.neuron_integrated_gradients import NeuronIntegratedGradients
from captum.attr._core.neuron_modified_relu_grad import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)

from .helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModel_ConvNet,
)
from .helpers.utils import BaseGPUTest


class Test(BaseGPUTest):
    def test_simple_input_internal_inf(self):
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
            InternalInfluence, net, net.relu, inputs=inp, target=0
        )

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

    def test_simple_layer_activation(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(LayerActivation, net, net.relu, inputs=inp)

    def test_multi_input_layer_activation(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerActivation,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
        )

    def test_simple_layer_conductance(self):
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
            LayerConductance, net, net.relu, inputs=inp, target=1
        )

    def test_multi_input_layer_conductance(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerConductance,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
            test_batches=True,
        )

    def test_multi_dim_layer_conductance(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerConductance, net, net.conv2, alt_device_ids=True, inputs=inp, target=1
        )

    def test_simple_layer_gradient_x_activation(self):
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
            LayerGradientXActivation, net, net.relu, inputs=inp, target=1
        )

    def test_multi_input_layer_gradient_x_activation(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerGradientXActivation,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
        )

    def test_multi_dim_layer_grad_cam(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerGradCam, net, net.conv2, alt_device_ids=True, inputs=inp, target=1
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
        if test_batches:
            batch_sizes = [None, 1, 8]
        for batch_size in batch_sizes:
            if batch_size:
                attributions_orig = attr_orig.attribute(
                    internal_batch_size=batch_size, **kwargs
                )
            else:
                attributions_orig = attr_orig.attribute(**kwargs)
            if batch_size:
                attributions_dp = attr_dp.attribute(
                    internal_batch_size=batch_size, **kwargs
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


if __name__ == "__main__":
    unittest.main()
