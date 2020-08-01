import warnings
from typing import Any, Iterable, List, Tuple, Union

import torch
from torch.nn import Module



def weighted_combine(
    saliency_maps: [Tensor, ...],
    weights: [float, ...],
    output_shape: Tuple, 
    resize_mode: str = "bilinear",
    relu_attribution: bool = False
) -> Tensor:
    
    assert len(saliency_maps) > 1, "need more than 1 saliency map to combine."
    assert len(weights) == len(saliency_maps), "weights and saliency maps \
        should have the same length."
    
    bn  = saliency_maps[0].size()[0]
    cm  = torch.zeros((bn, 1, output_shape[0], output_shape[1]), 
                        dtype=saliency_maps[0].dtype, 
                        device=saliency_maps[0].device)
    ww  = []
    for i, smap in enumerate(smaps):
        w = F.interpolate(smap.unsqueeze(1), 
                            size=output_shape, 
                            mode=resize_mode, 
                            align_corners=False) 
        ww.append(w)
        cm  += (w * weights[i])
    cm  = cm / np.sum(weights)
    cm  = cm.reshape(bn, output_shape[0], output_shape[1])
    ww  = torch.stack(ww,dim=1)
    ww  = ww.reshape(bn, len(weights), output_shape[0], output_shape[1])
    if relu_attribution:
        cm = F.relu(cm)
        ww = F.relu(ww)
    return cm, ww


class SaliencyMaskDropout(Module):
    def __init__(self, 
        keep_percent: float = 0.1, 
        return_layer_only: bool = False,
        scale_map: bool = True):
        super(SaliencyMaskDropout, self).__init__()
        
        assert 0 < keep_percent <= 1.0, "keep_percent should be a floating point value from 0 to 1"
        
        if scale_map:
            self.scale = 1.0 / keep_percent
        else:
            self.scale = 1.0
        self.keep_percent = keep_percent
        self.drop_percent = 1.0 - self.keep_percent
                
    def forward(self, image, saliency_map):
        assert torch.is_tensor(image), "image should be a Torch Tensor"
        assert torch.is_tensor(saliency_map), "saliency map should be a Torch Tensor"
        assert len(image.size()) == 4, "image should have dimensions (batch size, channels, height, width)"
        
        batch_size, channels, height, width = image.size()
        assert saliency_map.size() == (batch_size, height, width), "image and saliency maps should have same shape"
        
        smap = saliency_map.reshape(batch_size, height * width)
        num_samples = int(self.drop_percent * height * width)
        s = torch.sort(smap, dim=1)[0]

        assert s[:,0]  >= 0.0, "Saliency map should contain values within the range of 0 to 1"
        assert s[:,-1] <= 1.0, "Saliency map should contain values within the range of 0 to 1"
                
        # Get the kth value for each image in the batch.
        k = s[:,num_samples].reshape(batch_size, 1)
        
        # We will create the saliency mask but we use torch.autograd so that we can optionally
        # propagate the gradients backwards through the mask. k is assumed to be a dead-end, so 
        # no gradients go to it. 
        drop_map = torch.gt(smap, k)
        image = image.reshape(batch_size, channels, height * width)

        # Multiply the input by the mask, but optionally scale it like we would a dropout layer
        masked_image = image * drop_map.unsqueeze(1) * self.scale
        masked_image = masked_image.reshape(batch_size, channels, height, width)
        drop_map = drop_map.reshape(batch_size, height, width)
        return masked_image, drop_map
