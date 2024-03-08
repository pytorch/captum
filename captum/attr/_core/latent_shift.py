#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor


class LatentShift(GradientAttribution):
    r"""An implementation of the Latent Shift method to generate
    counterfactual explanations. This method uses an autoencoder to restrict
    the possible  adversarial examples to remain in the data space by
    adjusting the latent space of the autoencoder using dy/dz instead of
    dy/dx in order  to change the classifier's prediction.

    This class implements a search strategy to determine the lambda needed to
    change the prediction of the classifier by a specific amount as well  as
    the code to generate a video and construct a heatmap representing the
    image changes for viewing as an image.

    More details regarding the latent shift method can be found in the
    original paper:
    https://arxiv.org/abs/2102.09475
    And the original code repository:
    https://github.com/mlmed/gifsplanation
    """

    def __init__(self, forward_func: Callable, autoencoder) -> None:
        r"""
        Args:
            forward_func (callable): The forward function of the model or
                        any modification of it
            autoencoder: An object with an encode and decode function which
                        maintains a gradient tape.
        """
        GradientAttribution.__init__(self, forward_func)
        self.ae = autoencoder

        # check if ae has encode and decode
        assert hasattr(self.ae, "encode")
        assert hasattr(self.ae, "decode")

    @log_usage()
    def attribute(
        self,
        inputs: Tensor,
        target: int,
        fix_range: Union[Tuple, None] = None,
        search_pred_diff: float = 0.8,
        search_step_size: float = 10.0,
        search_max_steps: int = 3000,
        search_max_pixel_diff_pct: float = 0.05,
        lambda_sweep_steps: int = 10,
        heatmap_method: str = "int",
        apply_sigmoid: bool = True,
        verbose: bool = True,
        return_dicts: bool = False,
    ) -> Union[Tensor, List[Dict[str, Any]]]:
        r"""
        This method performs a search in order to determine the correct lambda
        values to generate the shift. The search starts by stepping by
        `search_step_size` in the negative direction while trying to determine
        if the output of the classifier has changed by `search_pred_diff` or
        when the change in the predict in stops going down. In order to avoid
        artifacts if the shift is too large or in the wrong direction an extra
        stop conditions is added `search_max_pixel_diff` if the change in the
        image is too large. To avoid the search from taking too long a
        `search_max_steps` will prevent the search from going on endlessly.


        Args:

            inputs (tensor):  Input for which the counterfactual is computed.
            target (int):  Output indices for which dydz is computed (for
                        classification cases, this is usually the target class).
            fix_range (tuple): Overrides searching and directly specifies the
                        lambda range to use. e.g. [-100,0].
            search_pred_diff (float): The desired change in the classifiers
                        prediction. For example if the classifer predicts 0.9
                        and pred_diff=0.8 the search will try to generate a
                        counterfactual where the prediction is 0.1.
            search_step_size (float): When searching for the right lambda to use
                        this will be the initial step size. This is similar to
                        a learning rate. Smaller values avoid jumping over the
                        ideal lambda but the search may take a long time.
            search_max_steps (int): The max steps to take when doing the search.
                        Sometimes steps make a tiny improvement and can go on
                        forever. This just bounds the time and gives up the
                        search.
            search_max_pixel_diff_pct (float): When searching, stop if the pixel
                        difference is larger than this amount. This will
                        prevent large  artifacts being introduced into the
                        image. |img0 - imgx| > |img0|*pct
            lambda_sweep_steps (int): How many frames to generate for the video.
            heatmap_method:  Default: 'int'. Possible methods: 'int': Average
                        per frame differences. 'mean' : Average difference
                        between 0 and other lambda frames. 'mm': Difference
                        between first and last frames. 'max': Max difference
                        from lambda 0 frame
            apply_sigmoid: Default: True. Apply a sigmoid to the output of the
                        model. Set to false to work with regression models or
                        if the model already applies a sigmoid.
            verbose: True to print debug text
            return_dicts (bool): Return a list of dicts containing information
                        from each image processed. Default False

        Returns:
            attributions or (if return_dict=True) a list of dicts containing the
                follow keys:
                generated_images: A list of images generated at each step along
                    the dydz vector from the smallest lambda to the largest. By
                    default the smallest lambda represents the counterfactual
                    image and the largest lambda is 0 (representing no change).
                lambdas: A list of the lambda values for each generated image.
                preds: A list of the predictions of the model for each generated
                    image.
                heatmap: A heatmap indicating the pixels which change in the
                    video sequence of images.


        Example::

            >>> # Load classifier and autoencoder
            >>> model = classifiers.FaceAttribute()
            >>> ae = autoencoders.VQGAN(weights="faceshq")
            >>>
            >>> # Load image
            >>> x = torch.randn(1, 3, 1024, 1024)
            >>>
            >>> # Defining Latent Shift module
            >>> attr = captum.attr.LatentShift(model, ae)
            >>>
            >>> # Computes counterfactual for class 3.
            >>> output = attr.attribute(x, target=3)

        """

        assert lambda_sweep_steps > 1, "lambda_sweep_steps must be at least 2"

        results = []
        # cheap batching
        for idx in range(inputs.shape[0]):
            inp = inputs[idx].unsqueeze(0)
            z = self.ae.encode(inp).detach()
            z.requires_grad = True
            x_lambda0 = self.ae.decode(z)
            pred = self.forward_func(x_lambda0)[:, target]
            if apply_sigmoid:
                pred = torch.sigmoid(pred)
            dzdxp = torch.autograd.grad(pred, z)[0]

            # Cache so we can reuse at sweep stage
            cache = {}

            def compute_shift(lambdax):
                """Compute the shift for a specific lambda"""
                if lambdax not in cache:
                    x_lambdax = self.ae.decode(z + dzdxp * lambdax).detach()
                    pred1 = self.forward_func(x_lambdax)[:, target]
                    if apply_sigmoid:
                        pred1 = torch.sigmoid(pred1)
                    pred1 = pred1.detach().cpu().numpy()
                    cache[lambdax] = x_lambdax, pred1
                return cache[lambdax]

            _, initial_pred = compute_shift(0)

            if fix_range:
                lbound, rbound = fix_range
            else:
                # Left range
                lbound = 0
                last_pred = initial_pred
                pixel_sum = x_lambda0.abs().sum()  # Used for pixel diff
                while True:
                    x_lambdax, cur_pred = compute_shift(lbound)
                    pixel_diff = torch.abs(x_lambda0 - x_lambdax).sum().detach().cpu()
                    if verbose:
                        toprint = [
                            f"Shift: {lbound}",
                            f"Pred: {float(cur_pred)}",
                            f"pixel_diff: {float(pixel_diff)}",
                            f"sum*diff_pct: {pixel_sum * search_max_pixel_diff_pct}",
                        ]
                        print(", ".join(toprint))

                    # If we stop decreasing the prediction
                    if last_pred < cur_pred:
                        break
                    # If the prediction becomes very low
                    if cur_pred < 0.05:
                        break
                    # If we have decreased the prediction by pred_diff
                    if initial_pred - search_pred_diff > cur_pred:
                        break
                    # If we are moving in the latent space too much
                    if lbound <= -search_max_steps:
                        break
                    # If we move too far we will distort the image
                    if pixel_diff > (pixel_sum * search_max_pixel_diff_pct):
                        break

                    last_pred = cur_pred
                    lbound = lbound - search_step_size + lbound // 10

                # Right range search not implemented
                rbound = 0

            if verbose:
                print("Selected bounds: ", lbound, rbound)

            # Sweep over the range of lambda values to create a sequence
            lambdas = np.linspace(lbound, rbound, lambda_sweep_steps)
            assert lambda_sweep_steps == len(
                lambdas
            ), "Inconsistent number of lambda steps"

            if verbose:
                print("Lambdas to compute: ", lambdas)

            preds = []
            generated_images = []

            for lam in lambdas:
                x_lambdax, pred = compute_shift(lam)
                generated_images.append(x_lambdax.cpu().numpy()[0])
                preds.append(float(pred))

            params = {}
            params["generated_images"] = np.array(generated_images)
            params["lambdas"] = lambdas
            params["preds"] = preds

            x_lambda0 = x_lambda0.detach().cpu().numpy()
            if heatmap_method == "max":
                # Max difference from lambda 0 frame
                heatmap = np.max(np.abs(x_lambda0 - generated_images), 0)

            elif heatmap_method == "mean":
                # Average difference between 0 and other lambda frames
                heatmap = np.mean(np.abs(x_lambda0 - generated_images), 0)

            elif heatmap_method == "mm":
                # Difference between first and last frames
                heatmap = np.abs(generated_images[0] - generated_images[-1])

            elif heatmap_method == "int":
                # Average per frame differences
                image_changes = []
                for i in range(len(generated_images) - 1):
                    image_changes.append(
                        np.abs(generated_images[i] - generated_images[i + 1])
                    )
                heatmap = np.mean(image_changes, 0)
            else:
                raise Exception("Unknown heatmap_method for 2d image")

            params["heatmap"] = heatmap
            results.append(params)

        if return_dicts:
            return results
        else:
            return torch.tensor([result["heatmap"] for result in results])
