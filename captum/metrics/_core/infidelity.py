import torch

from ..._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_input,
    _format_tensor_into_tuples,
    _is_tuple,
    _run_forward,
)


def infidelity_attr(
    forward_func,
    perturb_func,
    inputs,
    attributions,
    additional_forward_args=None,
    target=None,
    n_samples=10,
    max_examples_per_batch=None,
):
    r"""
    Explanation infidelity represents the expected mean-squared error
    between the explanation multiplied by a meaningful input perturbation
    and the differences between the predictor function at its input
    and perturbed input.
    More details about the measure can be found in the following paper:
    https://arxiv.org/pdf/1901.09392.pdf

    It is derived from the completeness property of well-known attribution
    algorithms and is an optimal and generalized notion of Sensitivy-n.
    The latter measures correlations between the sum of the attributions
    and the differences of the predictor function at its input and
    fixed baseline. More details about the Sensitivity-n can be found here:
    https://arxiv.org/pdf/1711.06104.pdfs

    The users can perturb the inputs any desired way by providing any
    perturbation function that takes the inputs and returns perturbed
    inputs and corresponding perturbations.

    This specific implementation is primarily tested for attribution-based
    explanation methods but the idea can be expanded to use for non
    attribution-based interpretability methods as well.

    Args:

        forward_func (callable):
                The forward function of our model or any modification of it.
        perturb_func (callable):
                The perturbation function of model inputs. This function takes
                model inputs as an argument and returns a tuple of perturbations
                and perturbeded inputs. If there is more than one input,
                corresponding perturbations must be computed and returned as tuples
                in the following format:
                    (perturb1, perturb2, ... perturbN), (input1, input2, ... inputN)
                In case inputs is a single tensor, the function can return a tuple
                of perturbations and inputs such as:
                    perturb, inputs

        inputs (tensor or tuple of tensors):  Input for which
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        attributions (tensor or tuple of tensors):
                Attribution scores computed based on an attribution algorithm.
                Attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.

        additional_forward_args (any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order, following the arguments in inputs.
                Note that the perturbations are not computed with respect
                to these arguments. This means that these arguments aren't
                being passed to `perturb_func` as an input argument.

                Default: None
        target (int, tuple, tensor or list, optional): Indices for selecting
                predictions from output(for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example, no target
                index is necessary.
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
        n_samples (int, optional): The number of times input tensors are perturbed.
                Each input example in the inputs tensor is expanded `n_samples`
                times before calling `perturb_func` function.

                Default: 10
        max_examples_per_batch (int, optional): The number of maximum input
                examples that are processed together. In case the number of
                examples exceeds `max_examples_per_batch`, they will be sliced
                into batches of `max_examples_per_batch` examples and processed
                in a sequential order. If `max_examples_per_batch` is None, all
                examples are processed together.

                Default: None

    Returns:

        infidelities (tensor): A tensor of scalar infidelity scores per
                        input example. The first dimension is equal to the
                        number of examples in the input batch and the second
                        dimension is one.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> dl = DeepLift(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes deeplift attribution scores for class 3.
        >>> attribution = dl.attribute(input, target=3)
        >>> # define a perturbation function for the input
        >>> def perturb_fn(inputs):
        >>>    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
        >>>    return noise, inputs - noise
        >>> # Computes infidelity score for deeplift attribution method
        >>> infidelity = infidelity_attr(net, perturb_fn, input, attribution)
    """

    def _generate_perturbations(inputs):
        inputs_expanded = tuple(
            torch.repeat_interleave(input, n_samples, dim=0) for input in inputs
        )
        return perturb_func(*inputs_expanded)

    def _partition_additional_forward_argument(start, end, additional_arg):
        if additional_arg is None:
            return None
        if not isinstance(additional_arg, torch.Tensor):
            return additional_arg
        assert bsz == additional_arg.size(0), (
            "If `additional_forward_arg` is a "
            "tensor, its first dimension must match with the input batch size"
        )

        return additional_arg[start:end]

    def _partition_target(start, end, target):
        if (
            not isinstance(target, (list, torch.Tensor))
            or (isinstance(target, torch.Tensor) and target.numel() == 1)
            or (bsz != len(target) and len(target) == 1)
        ):
            return target

        assert bsz == len(target), (
            "If `target` is a list or tensor of multiple elements, "
            "its length must match with the batch size"
        )

        return target[start:end]

    def _next_infidelity(start, end):
        if end <= start:
            return None

        inputs_partition = tuple(input[start:end] for input in inputs)
        perturbations, inputs_perturbed = _generate_perturbations(inputs_partition)

        if not is_input_tpl:
            perturbations = _format_tensor_into_tuples(perturbations)
            inputs_perturbed = _format_tensor_into_tuples(inputs_perturbed)

        # asserts the sizes of the perturbations and inputs
        assert len(perturbations) == len(inputs_perturbed), (
            """The number of perturbed
            inputs and corresponding perturbations must have the same number of
             elements. Found number of inputs is: {} and perturbations:
             {}"""
        ).format(len(perturbations), len(inputs_partition))

        # asserts the shapes of the perturbations and perturbed inputs
        for perturb, input_perturbed in zip(perturbations, inputs_perturbed):
            assert perturb.shape == input_perturbed.shape, (
                """Perturbed input
            and corresponding perturnbation must have the same shape and
            dimensionality. Found perturbation shape is: {} and the input shape
            is: {}"""
            ).format(perturb.shape, input_perturbed.shape)

        # partitions additional forward args before expanding
        additional_forward_args_partition = (
            None
            if additional_forward_args is None
            else tuple(
                _partition_additional_forward_argument(
                    start, end, additional_forward_arg
                )
                for additional_forward_arg in additional_forward_args
            )
        )
        # partitions target before expanding
        target_partition = _partition_target(start, end, target)
        inputs_fwd = _run_forward(
            forward_func,
            inputs_partition,
            target_partition,
            additional_forward_args_partition,
        )

        targets_expanded = _expand_target(
            target_partition, n_samples, expansion_type=ExpansionTypes.repeat_interleave
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args_partition,
            n_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        inputs_fwd = torch.repeat_interleave(inputs_fwd, n_samples, dim=0)
        inputs_minus_perturb = inputs_fwd - inputs_perturbed_fwd
        attributions_partition = tuple(
            torch.repeat_interleave(attribution[start:end], n_samples, dim=0)
            for attribution in attributions
        )
        attributions_times_perturb = tuple(
            (attribution_partition * perturbation).view(
                attribution_partition.size(0), -1
            )
            for attribution_partition, perturbation in zip(
                attributions_partition, perturbations
            )
        )
        attribution_times_perturb_sums = sum(
            [
                torch.sum(attribution_times_perturb, dim=1)
                for attribution_times_perturb in attributions_times_perturb
            ]
        )
        return torch.mean(
            torch.pow(attribution_times_perturb_sums - inputs_minus_perturb, 2).view(
                end - start, -1
            ),
            dim=1,
        )

    is_input_tpl = _is_tuple(inputs)
    inputs = _format_input(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    attributions = _format_tensor_into_tuples(attributions)

    # Make sure that inputs and corresponding attributions have matching sizes.
    assert len(inputs) == len(attributions), (
        """The number of tensors in the inputs and
        attributions must match. Found number of tensors in the inputs is: {} and in the
        attributions: {}"""
    ).format(len(inputs), len(attributions))
    for inp, attr in zip(inputs, attributions):
        assert inp.shape == attr.shape, (
            """Inputs and attributions must have
        matching shapes. One of the input tensor's shape is {} and the
        attribution tensor's shape is: {}"""
        ).format(inp.shape, attr.shape)

    bsz = inputs[0].size(0)
    max_inps_per_batch = (
        bsz if max_examples_per_batch is None else min(max_examples_per_batch, bsz)
    )

    with torch.no_grad():
        start = 0
        end = max_inps_per_batch

        infidelities = []
        infidelity = _next_infidelity(start, end)

        while infidelity is not None:
            infidelities.append(infidelity)
            start = end
            end = min(end + max_inps_per_batch, bsz)
            infidelity = _next_infidelity(start, end)
    return torch.cat(infidelities, dim=0)
