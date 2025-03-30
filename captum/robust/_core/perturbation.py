#!/usr/bin/env python3
from typing import Callable


class Perturbation:
    r"""
    All perturbation and attack algorithms extend this class. It enforces
    its child classes to extend and override core `perturb` method.
    """

    perturb: Callable
    r"""
    This method computes and returns the perturbed input for each input tensor.
    Deriving classes are responsible for implementing its logic accordingly.

    Specific adversarial attack algorithms that extend this class take relevant
    arguments.

    Args:

        inputs (Tensor or tuple[Tensor, ...]): Input for which adversarial attack
                    is computed. It can be provided as a single tensor or
                    a tuple of multiple tensors. If multiple input tensors
                    are provided, the batch sizes must be aligned across all
                    tensors.

    Returns:

        - **perturbed inputs** (*Tensor* or *tuple[Tensor, ...]*):
                    Perturbed input for each
                    input tensor. The perturbed inputs have the same shape and
                    dimensionality as the inputs.
                    If a single tensor is provided as inputs, a single tensor
                    is returned. If a tuple is provided for inputs, a tuple of
                    corresponding sized tensors is returned.
    """

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)
