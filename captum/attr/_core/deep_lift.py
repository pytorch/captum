#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from .._utils.common import (
    format_input,
    format_baseline,
    _format_attributions,
    validate_input,
)
from .._utils.attribution import GradientBasedAttribution


class DeepLift(GradientBasedAttribution):
    def __init__(self, model):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
        """
        super().__init__(model)
        self.model = model
        self.forward_handles = []
        self.backward_handles = []

    def attribute(
        self, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        r""""
            Rescale-rule implementation of deeplift approach based on
            the following paper.
            https://arxiv.org/pdf/1704.02685.pdf
            At this point we support one input sample with
            multiple references.

        Args:

            inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        mutliple input tensors are provided, the examples must
                        be aligned appropriately.
            target (int, optional):  Output index for which gradient is computed
                        (for classification cases, this is the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary. (Note: Tuples for multi
                        -dimensional output indices will be supported soon.)
            additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be a tuple containing tensors or
                        any arbitrary python types. These arguments are provided to
                        forward_func in order following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None


        Returns:

                attributions: with respect to each input features which has the
                              same shape and dimensionality as the input

        """
        """
            There might be multiple references per input. Match the references
            to inputs.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)
        baselines = format_baseline(baselines, inputs)

        self._traverse_modules(self.model, self._register_hooks, input_type="ref")
        self.forward_func(*baselines)
        multi_refs = [
            input.shape != baseline.shape for input, baseline in zip(inputs, baselines)
        ]

        # match the sizes of inputs and baselines in case of multiple references
        # for a single input
        inputs = tuple(
            input.repeat(
                [baseline.shape[0]] + [1] * (len(baseline.shape) - 1)
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )
        validate_input(inputs, baselines)

        self._traverse_modules(self.model, self._register_hooks)
        gradients = self.gradient_func(
            self.forward_func,
            inputs,
            target_ind=target,
            additional_forward_args=additional_forward_args,
        )
        attributions = tuple(
            (input - baseline) * gradient
            for input, baseline, gradient in zip(inputs, baselines, gradients)
        )

        # remove hooks from all activations
        self._remove_hooks()

        # Average for multiple references
        attributions = tuple(
            torch.mean(result, axis=0, keepdim=False) if multi_ref else result
            for multi_ref, result in zip(multi_refs, attributions)
        )
        return _format_attributions(is_inputs_tuple, attributions)

    def _is_non_linear(self, module):
        module_name = module._get_name()
        return module_name in SUPPORTED_NON_LINEAR.keys()

    # we need forward hook to access and detach the inputs and outputs of a neuron
    def _forward_hook(self, module, inputs, outputs):
        input_attr_name = "input"
        output_attr_name = "output"
        self._detach_tensors(input_attr_name, output_attr_name, module, inputs, outputs)

    def _forward_hook_ref(self, module, inputs, outputs):
        input_attr_name = "input_ref"
        output_attr_name = "output_ref"
        self._detach_tensors(input_attr_name, output_attr_name, module, inputs, outputs)
        # if it is a reference forward hook remove it from the module after
        # detaching the variables
        module.ref_handle.remove()
        del module.ref_handle

    def _detach_tensors(
        self, input_attr_name, output_attr_name, module, inputs, outputs
    ):
        setattr(module, input_attr_name, tuple(input.detach() for input in inputs))
        setattr(module, output_attr_name, tuple(output.detach() for output in outputs))

    def _backward_hook(self, module, grad_input, grad_output, eps=1e-6):
        r"""
         grad_input: (dLoss / dlayer_out, )
         grad_output: (dLoss / dprev_layer_out, dLoss / wij, dLoss / bij)
         https://github.com/pytorch/pytorch/issues/12331
         """
        delta_in = tuple(
            inp - inp_ref for inp, inp_ref in zip(module.input, module.input_ref)
        )
        delta_out = tuple(
            out - out_ref for out, out_ref in zip(module.output, module.output_ref)
        )

        modified_grads = [g_input for g_input in grad_input]

        """
         `grad_input` is the gradient of the neuron with respect to its input
         `grad_output` is the gradient of the neuron with respect to its output
         we can override `grad_input` according to chain rule with
         `grad_output` * delta_out/ delta_in.
        """

        # del module.input
        # del module.output
        return tuple(
            SUPPORTED_NON_LINEAR[module._get_name()](
                module, delta_in, delta_out, modified_grads, grad_output, eps=1e-6
            )
        )

    def _register_hooks(self, module, input_type="non_ref"):
        # TODO find a better way of checking if a module is a container or not
        module_fullname = str(type(module))
        has_already_hooks = len(module._backward_hooks) > 0
        if (
            "nn.modules.container" in module_fullname
            or has_already_hooks
            or not self._is_non_linear(module)
        ):
            return
        # adds forward hook to leaf nodes that are non-linear
        if input_type != "ref":
            forward_handle = module.register_forward_hook(self._forward_hook)
            backward_handle = module.register_backward_hook(self._backward_hook)
            self.forward_handles.append(forward_handle)
            self.backward_handles.append(backward_handle)
        else:
            handle = module.register_forward_hook(self._forward_hook_ref)
            ref_handle = "ref_handle"
            setattr(module, ref_handle, handle)

    def _traverse_modules(self, module, hook_fn, input_type="non_ref"):
        children = module.children()
        for child in children:
            self._traverse_modules(child, hook_fn, input_type)
            hook_fn(child, input_type)

    def _remove_hooks(self):
        for forward_handle in self.forward_handles:
            forward_handle.remove()
        for backward_handle in self.backward_handles:
            backward_handle.remove()


def nonlinear(module, delta_in, delta_out, grad_input, grad_output, eps=1e-6):
    """
         grad_input: (dLoss / dlayer_out, )
         grad_output: (dLoss / dprev_layer_out, dLoss / wij, dLoss / bij)
         https://github.com/pytorch/pytorch/issues/12331
         """
    # supported non-linear modules take only single tensor as input hence accessing
    # only the first element in `grad_input` and `grad_output`

    grad_input[0] = torch.where(
        delta_in[0] < eps, grad_input[0], grad_output[0] * delta_out[0] / delta_in[0]
    )
    return grad_input


def maxpool1d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-6):
    return maxpool(
        module,
        F.max_pool1d,
        F.max_unpool1d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool2d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-6):
    return maxpool(
        module,
        F.max_pool2d,
        F.max_unpool2d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool3d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-6):
    return maxpool(
        module,
        F.max_pool3d,
        F.max_unpool3d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def softmax(module, delta_in, delta_out, grad_input, grad_output, eps=1e-6):
    grad_input_unnorm = torch.where(
        delta_in[0] < eps, grad_input[0], grad_output[0] * delta_out[0] / delta_in[0]
    )
    # normalizing
    n = grad_input[0].shape[1]
    grad_input[0] = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return grad_input


def maxpool(
    module,
    pool_func,
    unpool_func,
    delta_in,
    delta_out,
    grad_input,
    grad_output,
    eps=1e-6,
):
    # The forward function of maxpool takes only tensors not
    # a tuple hence accessing the first
    # element in the tuple of inputs, grad_input and grad_output
    _, indices = pool_func(
        module.input[0],
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        module.ceil_mode,
        True,
    )
    unpool_grad_out_delta = unpool_func(
        grad_output[0] * delta_out[0],
        indices,
        module.kernel_size,
        module.stride,
        module.padding,
        list(module.input[0].shape),
    )

    grad_input[0] = torch.where(
        delta_in[0] < eps, grad_input[0], unpool_grad_out_delta / delta_in[0]
    )
    return grad_input


SUPPORTED_NON_LINEAR = {
    "ReLU": nonlinear,
    "Elu": nonlinear,
    "LeakyReLU": nonlinear,
    "Sigmoid": nonlinear,
    "Tanh": nonlinear,
    "Softplus": nonlinear,
    "MaxPool1d": maxpool1d,
    "MaxPool2d": maxpool2d,
    "MaxPool3d": maxpool3d,
    "Softmax": softmax,
}
