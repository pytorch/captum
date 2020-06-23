import torch

# TODO: use imageio to redo load and avoid TF dependency
from lucid.misc.io import load

DOG_CAT_URL = (
    "https://lucid-static.storage.googleapis.com/building-blocks/examples/dog_cat.png"
)


def image(url: str = DOG_CAT_URL):
    img_np = load(url)
    return torch.as_tensor(img_np.transpose(2, 0, 1))
