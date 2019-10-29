#!/usr/bin/env python3

import torch

from .._utils.common import _format_attributions, format_input, _run_forward
from .._utils.attribution import GradientAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


class Occlusion(PerturbationAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        super().__init__(forward_func)

    def attribute(self, inputs, target=None, additional_forward_args=None, feature_mask=None, baselines=0, _internal_batch_size=None):
        r""""
        A baseline approach for computing input attribution. It returns
        the gradients with respect to inputs. If `abs` is set to True, which is
        the default, the absolute value of the gradients is returned.

        More details about the approach can be found in the following paper:
            https://arxiv.org/pdf/1312.6034.pdf

        Args:

                inputs (tensor or tuple of tensors):  Input for which integrated
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            multiple input tensors are provided, the examples must
                            be aligned appropriately.
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
                abs (bool, optional): Returns absolute value of gradients if set
                            to True, otherwise returns the (signed) gradients if
                            False.
                            Defalut: True
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be either a single additional
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None

        Returns:
                *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                            The gradients with respect to each input feature.
                            Attributions will always be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # Generating random input with size 2x3x3x32
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Defining Saliency interpreter
            >>> saliency = Saliency(net)
            >>> # Computes saliency maps for class 3.
            >>> attribution = saliency.attribute(input, target=3)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = format_input(inputs)
        num_examples = inputs[0].shape[0]
        feature_mask = format_input(feature_mask) if feature_mask is not None else None
        assert internal_batch_size >= inputs[0].shape[0], "Internal batch size must be at least number of examples given."
        initial_eval = _run_forward(self.forward_func, inputs, target, additional_forward_args)
        assert initial_eval[0].numel() == 1, "Target should identify a single element in the model output."

        total_attrib = [torch.zeros_like(input) for input in inputs]
        weights = [torch.zeros_like(input[0:1]) for input in inputs]

        for i in range(len(inputs)):
            input_mask = torch.reshape(torch.arange(torch.numel(inputs[i][0])), inputs[i][0:1].shape) if feature_mask is None else feature_mask[i]
            num_features = torch.max(input_mask).item() + 1
            batch_size = num_features if internal_batch_size is None else min(internal_batch_size // inputs[0].shape[0], num_features)
            baseline = baselines[i] if isinstance(baselines, tuple) else baselines
            all_features_repeated = [torch.cat([inputs[j]] * batch_size, dim=0) for j in range(len(inputs))]
            evals_repeated = torch.cat([initial_eval] * batch_size, dim=0)

            num_features_processed = 0
            while num_features_processed < num_features:
                current_batch = min(batch_size, num_features - num_features_processed)
                current_mask = torch.cat([input_mask == j for j in range(num_features_processed, num_features_processed + current_batch)], dim=0)
                weights[i] += current_mask.float().sum(dim=0)
                current_mask = current_mask.repeat_interleave(inputs[i].shape[0], dim=0).long()
                if current_batch != batch_size:
                    current_features = [tensor[0:current_batch * num_examples] for tensor in all_features_repeated]
                    current_evals = evals_repeated[0:current_batch * num_examples]
                else:
                    current_features = all_features_repeated
                    current_evals = evals_repeated
                original_tensor = current_features[i]
                current_features[i] = (current_features[i] * (1 - current_mask).float()) + (baseline * current_mask.float())
                modified_eval = _run_forward(self.forward_func, tuple(current_features), target, additional_forward_args)
                evals = (current_evals - modified_eval).reshape((len(modified_eval),) + (len(inputs[i].shape) - 1) * (1,))

                total_attrib[i] += (evals * current_mask.float()).reshape((current_batch, num_examples) + inputs[i].shape[1:]).sum(dim=0)
                current_features[i] = original_tensor
                num_features_processed += current_batch
        div_attrib = tuple(single_attrib / weight for single_attrib, weight in zip(total_attrib, weights))
        return _format_attributions(is_inputs_tuple, div_attrib)
