#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from .._utils.attribution import GradientAttribution
from .._utils.common import format_input, _format_attributions
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


class GuidedBackprop(GradientAttribution):
    def __init__(self, model):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
        """
        super().__init__(model)
        self.model = model
        self.backward_hooks = []
        assert isinstance(self.model, torch.nn.Module), (
            "Given model must be an instance of torch.nn.Module to properly hook"
            " ReLU layers."
        )

    def attribute(self, inputs, target=None, additional_forward_args=None):
        r""""
        Computes attribution using guided backpropagation. Guided backpropagation
        computes the gradient of the target output with respect the input,
        but gradients of ReLU functions are overriden so that only
        non-negative gradients are backpropagated.

        More details regarding the guided backpropagation algorithm can be found
        in the original paper here:
        https://arxiv.org/abs/1412.6806

        Warning: Ensure that all ReLU operations in the forward function of the
        given model are performed using a module (nn.module.ReLU).
        If nn.functional.ReLU is used, gradients are not overriden appropriately.

        Args:

            inputs (tensor or tuple of tensors):  Input for which
                        attributions are computed. If forward_func takes a single
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
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
        Returns:

            attributions (tensor or tuple of tensors): The guided backprop gradients
                        with respect to each input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> gbp = GuidedBackprop(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes Guided Backprop attribution scores for class 3.
            >>> attribution, delta = gbp.attribute(input, target=3)
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # set hooks for overriding ReLU gradients
        self.model.apply(self._register_hooks)

        gradients = self.gradient_func(
            self.forward_func, inputs, target, additional_forward_args
        )

        # remove set hooks
        self._remove_hooks()

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, gradients)

    def _register_hooks(self, module):
        if isinstance(module, torch.nn.ReLU):
            hook = module.register_backward_hook(self._backward_hook)
            self.backward_hooks.append(hook)

    def _backward_hook(self, module, grad_input, grad_output):
        if isinstance(grad_input, tuple):
            return tuple(F.relu(inp) for inp in grad_input)
        else:
            return F.relu(grad_input)

    def _remove_hooks(self):
        for hook in self.backward_hooks:
            hook.remove()
