#!/usr/bin/env python3

from typing import Any, Callable, Tuple

import torch
from torch import Tensor
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage


class LatentShift(GradientAttribution):
    r"""An implementation of the Latent Shift method to generate 
    counterfactual explainations. This method uses an audoencoder to restrict
    the possible  adverserial examples to remain in the dataspace by adjusting 
    the latent space of the autoencoder using dy/dz instead of dy/dx in order 
    to change the classifier's prediction.
    
    This class implements a search strategy to determine the lambda needed 
    to change the prediction of the classifier by a specific amount as well
    as the code to generate a video and construct a heatmap representing 
    the image changes for viewing as an image.
    
    Publication:
    Cohen, J. P., et al. Gifsplanation via Latent Shift: A Simple Autoencoder 
    Approach to Counterfactual Generation for Chest X-rays. Medical Imaging 
    with Deep Learning. https://arxiv.org/abs/2102.09475
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
        target: TargetType = None,
        fix_range: Tuple = None,
        pred_diff: float = 0.8,
        shift_step_size: float = 10.0,
        max_pixel_diff: float = 5000.0,
        aggregrate_method: str = 'int',
    ) -> dict:
        r"""
        Args:

            inputs (tensor):  Input for which the counterfactual is computed.
            target (int):  Output indices for which dydz is computed (for
                        classification cases, this is usually the target class).
            fix_range (tuple): Overrides searching and directly specifies the
                        lambda range to use. e.g. [-100,0].
            pred_diff (float): The desired change in the classifiers prediction.
                        For example if the classifer predicts 0.9 and 
                        pred_diff=0.8 the search will try to generate a  
                        counterfactual where the prediction is 0.1. 
            shift_step_size (float): When searching for the right lambda to use 
                        this will be the initial step size. This is similar to 
                        a learning rate. Smaller values avoid jumping over the 
                        ideal lambda but the search may take a long time.
            max_pixel_diff (float): When searching stop of the pixel difference
                        is larger than this amount. This will prevent large 
                        artifacts being introduced into the image.
            aggregate_method:  Default: 'int'. Possible methods: 'int': Average 
                        per frame differences. 'mean' : Average difference 
                        between 0 and other lambda frames. 'mm': Difference 
                        between first and last frames. 'max': Max difference 
                        from lambda 0 frame

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


        Examples::

            >>> # Load classifier and autoencoder
            >>> model = classifiers.FaceAttribute()
            >>> ae = autoencoders.Transformer(weights="celeba")
            >>> 
            >>> # Load image
            >>> input = torch.randn(1, 3, 1024, 1024)
            >>> 
            >>> # Defining Saliency interpreter
            >>> attr = captum.attr.LatentShift(model, ae)
            >>> 
            >>> # Computes counterfactual for class 3.
            >>> output = attr.attribute(input, target=3)

        """
        z = self.ae.encode(inputs).detach()
        z.requires_grad = True
        x_lambda0 = self.ae.decode(z)
        pred = torch.sigmoid(self.forward_func(x_lambda0))[:,target]
        dzdxp = torch.autograd.grad((pred), z)[0]
        
        # Cache so we can reuse at sweep stage
        cache = {}
        def compute_shift(lam):
            """Compute the shift for a specific lambda"""
            if lam not in cache:
                x_lambdax = self.ae.decode(z+dzdxp*lam).detach()
                pred1 = torch.sigmoid(self.forward_func(x_lambdax))[:,target]
                pred1 = pred1.detach().cpu().numpy() 
                cache[lam] = x_lambdax, pred1
                print(f'Shift: {lam} , Prediction: {pred1}')
            return cache[lam]
        
        _, initial_pred = compute_shift(0)
        
        if fix_range:
            lbound, rbound = fix_range
        else:
            # Left range
            lbound = 0
            last_pred = initial_pred
            while True:
                x_lambdax, cur_pred = compute_shift(lbound)
                pixel_diff = torch.abs(x_lambda0-x_lambdax).sum().detach()
                
                # If we stop decreasing the prediction
                if last_pred < cur_pred:
                    break
                # If the prediction becomes very low
                if cur_pred < 0.05:
                    break
                # If we have decreased the prediction by pred_diff
                if initial_pred - pred_diff > cur_pred:
                    break
                # If we are moving in the latent space too much
                if lbound <= -3000:
                    break
                # If we move too far we will distort the image
                if pixel_diff > max_pixel_diff:
                    break
                    
                last_pred = cur_pred
                lbound = lbound - shift_step_size + lbound//10

            # Right range search not implemented
            rbound = 0
        
        print('Selected bounds: ', lbound, rbound)
        
        # Sweep over the range of lambda values to create a sequence
        lambdas = np.arange(lbound, rbound, np.abs((lbound-rbound)/lambda_sweep_steps))
        
        preds = []
        generated_images = []
        
        for lam in lambdas:
            x_lambdax, pred = compute_shift(lam)
            generated_images.append(x_lambdax.cpu().numpy())
            preds.append(pred)
            
        params = {}
        params['generated_images'] = generated_images
        params['lambdas'] = lambdas
        params['preds'] = preds
        
        
        x_lambda0 = x_lambda0.detach().cpu().numpy()
        if heatmap_method == 'max':
            # Max difference from lambda 0 frame
            heatmap = np.max(np.abs(x_lambda0[0][0] - generated_images[0][0]),0)
        
        elif heatmap_method == 'mean':
            # Average difference between 0 and other lambda frames
            heatmap = np.mean(np.abs(x_lambda0[0][0] - generated_images[0][0]),0)
        
        elif heatmap_method == 'mm':
            # Difference between first and last frames
            heatmap = np.abs(generated_images[0][0][0] - generated_images[-1][0][0])
        
        elif heatmap_method == 'int':
            # Average per frame differences 
            image_changes = []
            for i in range(len(generated_images)-1):
                image_changes.append(np.abs(generated_images[i][0][0] - generated_images[i+1][0][0]))
            heatmap = np.mean(image_changes, 0)
        else:
            raise Exception('Unknown heatmap_method for 2d image')
            
        params["heatmap"] = heatmap
        
        return params
