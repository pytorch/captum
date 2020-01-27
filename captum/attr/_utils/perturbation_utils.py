#!/usr/bin/env python3

import torch


def _find_output_mode_and_verify(
    initial_eval, num_examples, ablations_per_eval, feature_mask
):
    """
    This method identifies whether the model outputs a single output for a batch
    (single_output_mode = True) or whether it outputs a single output per example
    (single_output_mode = False) and returns single_output_mode. The method also
    verifies that ablations_per_eval is 1 in the case that single_output_mode is True
    and also verifies that the first dimension of each feature mask if the model
    returns a single output for a batch.
    """
    if isinstance(initial_eval, (int, float)) or (
        isinstance(initial_eval, torch.Tensor)
        and (
            len(initial_eval.shape) == 0
            or (num_examples > 1 and initial_eval.numel() == 1)
        )
    ):
        single_output_mode = True
        assert (
            ablations_per_eval == 1
        ), "Cannot have ablations_per_eval > 1 when function returns scalar."
        if feature_mask is not None:
            for single_mask in feature_mask:
                assert (
                    single_mask.shape[0] == 1
                ), "Cannot provide multiple masks when function returns a scalar."
    else:
        single_output_mode = False
        assert (
            isinstance(initial_eval, torch.Tensor) and initial_eval[0].numel() == 1
        ), "Target should identify a single element in the model output."
    return single_output_mode
