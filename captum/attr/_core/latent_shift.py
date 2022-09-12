#!/usr/bin/env python3

from typing import Callable, Tuple, List

import os
import shutil
import torch
import numpy as np
from torch import Tensor
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
import subprocess
import matplotlib
import matplotlib.pyplot as plt


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

    Publication:
    Cohen, J. P., et al. Gifsplanation via Latent Shift: A Simple
    Autoencoder Approach to Counterfactual Generation for Chest
    X-rays. Medical Imaging with Deep Learning.
    https://arxiv.org/abs/2102.09475
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
        assert hasattr(self.ae, 'encode')
        assert hasattr(self.ae, 'decode')

    @log_usage()
    def attribute(
        self,
        inputs: Tensor,
        target: int,
        fix_range: Tuple = None,
        search_pred_diff: float = 0.8,
        search_step_size: float = 10.0,
        search_max_steps: int = 3000,
        search_max_pixel_diff: float = 5000.0,
        lambda_sweep_steps: int = 10,
        heatmap_method: str = 'int',
        verbose: bool = True,
    ) -> dict:
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
            search_max_pixel_diff (float): When searching stop of the pixel
                        difference is larger than this amount. This will
                        prevent large  artifacts being introduced into the
                        image.
            lambda_sweep_steps (int): How many frames to generate for the video.
            heatmap_method:  Default: 'int'. Possible methods: 'int': Average
                        per frame differences. 'mean' : Average difference
                        between 0 and other lambda frames. 'mm': Difference
                        between first and last frames. 'max': Max difference
                        from lambda 0 frame
            verbose: True to print debug text

        Returns:
            dict containing the follow keys:
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
            >>> ae = autoencoders.Transformer(weights="faceshq")
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
        z = self.ae.encode(inputs).detach()
        z.requires_grad = True
        x_lambda0 = self.ae.decode(z)
        pred = torch.sigmoid(self.forward_func(x_lambda0))[:, target]
        dzdxp = torch.autograd.grad(pred, z)[0]

        # Cache so we can reuse at sweep stage
        cache = {}

        def compute_shift(lambdax):
            """Compute the shift for a specific lambda"""
            if lambdax not in cache:
                x_lambdax = self.ae.decode(z + dzdxp * lambdax).detach()
                pred1 = torch.sigmoid(self.forward_func(x_lambdax))[:, target]
                pred1 = pred1.detach().cpu().numpy()
                cache[lambdax] = x_lambdax, pred1
                if verbose:
                    print(f'Shift: {lambdax} , Prediction: {pred1}')
            return cache[lambdax]

        _, initial_pred = compute_shift(0)

        if fix_range:
            lbound, rbound = fix_range
        else:
            # Left range
            lbound = 0
            last_pred = initial_pred
            while True:
                x_lambdax, cur_pred = compute_shift(lbound)
                pixel_diff = torch.abs(x_lambda0 - x_lambdax).sum().detach()

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
                if pixel_diff > search_max_pixel_diff:
                    break

                last_pred = cur_pred
                lbound = lbound - search_step_size + lbound // 10

            # Right range search not implemented
            rbound = 0

        if verbose:
            print('Selected bounds: ', lbound, rbound)

        # Sweep over the range of lambda values to create a sequence
        lambdas = np.arange(
            lbound,
            rbound,
            np.abs((lbound - rbound) / lambda_sweep_steps)
        ).tolist()

        preds = []
        generated_images = []

        for lam in lambdas:
            x_lambdax, pred = compute_shift(lam)
            generated_images.append(x_lambdax.cpu().numpy())
            preds.append(float(pred))

        params = {}
        params['generated_images'] = generated_images
        params['lambdas'] = lambdas
        params['preds'] = preds

        x_lambda0 = x_lambda0.detach().cpu().numpy()
        if heatmap_method == 'max':
            # Max difference from lambda 0 frame
            heatmap = np.max(
                np.abs(x_lambda0[0][0] - generated_images[0][0]), 0
            )

        elif heatmap_method == 'mean':
            # Average difference between 0 and other lambda frames
            heatmap = np.mean(
                np.abs(x_lambda0[0][0] - generated_images[0][0]), 0
            )

        elif heatmap_method == 'mm':
            # Difference between first and last frames
            heatmap = np.abs(
                generated_images[0][0][0] - generated_images[-1][0][0]
            )

        elif heatmap_method == 'int':
            # Average per frame differences
            image_changes = []
            for i in range(len(generated_images) - 1):
                image_changes.append(np.abs(
                    generated_images[i][0][0] - generated_images[i + 1][0][0]
                ))
            heatmap = np.mean(image_changes, 0)
        else:
            raise Exception('Unknown heatmap_method for 2d image')

        params["heatmap"] = heatmap

        return params

    @log_usage()
    def generate_video(
        self,
        params: dict,
        target_filename: str,
        watermark: bool = True,
        ffmpeg_path: str = "ffmpeg",
        temp_path: str = "/tmp/gifsplanation",
        show: bool = True,
        verbose: bool = True,
    ):
        """Generate a video from the generated images.

            Args:
                params: The dict returned from the call to `attribute`.
                target_filename: The filename to write the video to. `.mp4` will
                    be added to the end of the string.
                watermark: To add the probability output and the name of the
                    method.
                ffmpeg_path: The path to call `ffmpeg`
                temp_path: A temp path to write images.
                show: To try and show the video in a jupyter notebook.
                verbose: True to print debug text

            Returns:
                The filename of the video if show=False, otherwise it will
                return a video to show in a jupyter notebook.
        """

        if os.path.exists(target_filename + ".mp4"):
            os.remove(target_filename + ".mp4")

        shutil.rmtree(temp_path, ignore_errors=True)
        os.mkdir(temp_path)

        imgs = [h.transpose(0, 2, 3, 1) for h in params["generated_images"]]

        # Add reversed so we have an animation cycle
        towrite = list(reversed(imgs)) + list(imgs)
        ys = list(reversed(params['preds'])) + list(params['preds'])

        for idx, img in enumerate(towrite):

            px = 1 / plt.rcParams['figure.dpi']
            full_frame(img[0].shape[0] * px, img[0].shape[1] * px)
            plt.imshow(img[0], interpolation='none')

            if watermark:
                # Write prob output in upper left
                plt.text(
                    0.05, 0.95, f"{float(ys[idx]):1.1f}",
                    ha='left', va='top',
                    transform=plt.gca().transAxes
                )
                # Write method name in lower right
                plt.text(
                    0.96, 0.1, 'gifsplanation',
                    ha='right', va='bottom',
                    transform=plt.gca().transAxes
                )

            plt.savefig(
                f'{temp_path}/image-{idx}.png',
                bbox_inches='tight',
                pad_inches=0,
                transparent=False
            )
            plt.close()

        # Command for ffmpeg to generate an mp4
        cmd = f"{ffmpeg_path} -loglevel quiet -stats -y " \
              f"-i {temp_path}/image-%d.png " \
              f"-c:v libx264 -vf scale=-2:{imgs[0][0].shape[0]} " \
              f"-profile:v baseline -level 3.0 -pix_fmt yuv420p " \
              f"'{target_filename}.mp4'"

        if verbose:
            print(cmd)
        output = subprocess.check_output(cmd, shell=True)
        if verbose:
            print(output)

        if show:
            # If we in a jupyter notebook then show the video.
            from IPython.core.display import Video
            try:
                return Video(target_filename + ".mp4",
                             html_attributes="controls loop autoplay muted",
                             embed=True,
                             )
            except TypeError:
                return Video(target_filename + ".mp4",
                             embed=True
                             )
        else:
            return target_filename + ".mp4"


def full_frame(width=None, height=None):
    """Setup matplotlib so we can write to the entire canvas"""

    matplotlib.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
