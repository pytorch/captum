from io import BytesIO

import torch
from torchvision import transforms

from IPython import display, get_ipython


def tensor_jpeg(tensor: torch.Tensor):
    if tensor.dim() == 3:
        pil_image = transforms.ToPILImage()(tensor.cpu().detach()).convert("RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="jpeg")
        data = buffer.getvalue()
        return data
    else:
        return tensor


def register_formatters():
    jpeg_formatter = get_ipython().display_formatter.formatters["image/jpeg"]
    jpeg_formatter.for_type(torch.Tensor, tensor_jpeg)
