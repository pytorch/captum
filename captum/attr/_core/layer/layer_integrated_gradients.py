#!/usr/bin/env python3
from typing import Callable, List, Optional, Tuple, Union, Any
import torch
from torch import Tensor

from torch.nn import Module
from torch.nn.parallel.scatter_gather import scatter

from captum.attr._utils.common import (
    _tensorize_baseline,
    _validate_input,
    _format_additional_forward_args,
    _format_attributions,
    _format_input_baseline,
    _extract_device,
)

from captum.attr._utils.gradient import _forward_layer_eval

from captum.attr._utils.attribution import LayerAttribution, GradientAttribution
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.gradient import _run_forward


class LayerIntegratedGradients(LayerAttribution, GradientAttribution):
    r"""
    Layer Integrated Gradients is a variant of Integrated Gradients that assigns
    an importance score to layer inputs or outputs, depending on whether we
    attribute to the former or to the latter one.

    Integrated Gradients is an axiomatic model interpretability algorithm that
    attributes / assigns an importance score to each input feature by approximating
    the integral of gradients of the model's output with respect to the inputs
    along the path (straight line) from given baselines / references to inputs.

    Baselines can be provided as input arguments to attribute method.
    To approximate the integral we can choose to use either a variant of
    Riemann sum or Gauss-Legendre quadrature rule.

    More details regarding the integrated gradients method can be found in the
    original paper:
    https://arxiv.org/abs/1703.01365

    """

    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        r"""
        Args:
            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          the attribution of each neuron in the input or output
                          of this layer.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.

        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        GradientAttribution.__init__(self, forward_func)
        self.ig = IntegratedGradients(forward_func)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Optional[
            Union[Tensor, int, float, Tuple[Union[Tensor, int, float], ...]]
        ] = None,
        target: Optional[
            Union[int, Tuple[int, ...], Tensor, List[Tuple[int, ...]]]
        ] = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Optional[int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[
        Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to layer inputs or outputs of the model, depending on whether
        `attribute_to_layer_input` is set to True or False, using the approach
        described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:

            inputs (tensor or tuple of tensors):  Input for which layer integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                            exactly the same dimensions as inputs or the first
                            dimension is one and the remaining dimensions match
                            with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                                be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                            to each tensor in the inputs' tuple can be:
                            - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding
                                input tensor.
                            - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted
                                for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                            integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                            the number of examples in inputs (dim 0). Each integer
                            is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                            elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                            examples in inputs (dim 0), and each tuple containing
                            #output_dims - 1 elements. Each tuple is applied as the
                            target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (string, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            Returns:
                **attributions** or 2-element tuple of **attributions**, **delta**:
                - **attributions** (*tensor* or tuple of *tensors*):
                        Integrated gradients with respect to `layer`'s inputs or
                        outputs. Attributions will always be the same size and
                        dimensionality as the input or output of the given layer,
                        depending on whether we attribute to the inputs or outputs
                        of the layer which is decided by the input flag
                        `attribute_to_layer_input`.
                - **delta** (*tensor*, returned if return_convergence_delta=True):
                        The difference between the total approximated and true
                        integrated gradients. This is computed using the property
                        that the total sum of forward_func(inputs) -
                        forward_func(baselines) must equal the total sum of the
                        integrated gradient.
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in inputs.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx12x32x32.
                >>> net = ImageClassifier()
                >>> lig = LayerIntegratedGradients(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes layer integrated gradients for class 3.
                >>> # attribution size matches layer output, Nx12x32x32
                >>> attribution = lig.attribute(input, target=3)
        """
        inps, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inps, baselines, n_steps, method)

        baselines = _tensorize_baseline(inps, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)
        inputs_layer, is_layer_tuple = _forward_layer_eval(
            self.forward_func,
            inps,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        baselines_layer, _ = _forward_layer_eval(
            self.forward_func,
            baselines,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        # inputs -> these inputs are scaled
        def gradient_func(
            forward_fn: Callable,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target_ind: Optional[
                Union[int, Tuple[int, ...], Tensor, List[Tuple[int, ...]]]
            ] = None,
            additional_forward_args: Any = None,
        ) -> Tuple[Tensor, ...]:
            if self.device_ids is None:
                scattered_inputs = (inputs,)
            else:
                # scatter method does not have a precise enough return type in its
                # stub, so suppress the type warning.
                scattered_inputs = scatter(  # type:ignore
                    inputs, target_gpus=self.device_ids
                )

            scattered_inputs_dict = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }

            with torch.autograd.set_grad_enabled(True):

                def layer_forward_hook(module, hook_inputs, hook_outputs=None):
                    device = _extract_device(module, hook_inputs, hook_outputs)
                    if is_layer_tuple:
                        return scattered_inputs_dict[device]
                    return scattered_inputs_dict[device][0]

                if attribute_to_layer_input:
                    hook = self.layer.register_forward_pre_hook(layer_forward_hook)
                else:
                    hook = self.layer.register_forward_hook(layer_forward_hook)

                output = _run_forward(
                    self.forward_func, additional_forward_args, target_ind,
                )
                hook.remove()
                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(torch.unbind(output), inputs)
            return grads

        self.ig.gradient_func = gradient_func
        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )
        attributions = self.ig.attribute(
            inputs_layer,
            baselines=baselines_layer,
            target=target,
            additional_forward_args=all_inputs,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=False,
        )

        if return_convergence_delta:
            start_point, end_point = baselines, inps
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_attributions(is_layer_tuple, attributions), delta
        return _format_attributions(is_layer_tuple, attributions)

    def has_convergence_delta(self) -> bool:
        return True
