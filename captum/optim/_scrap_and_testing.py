import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import requests
from PIL import Image
from IPython.display import display

from clarity.pytorch.inception_v1 import googlenet
from lucid.misc.io import show, load, save
from lucid.modelzoo.other_models import InceptionV1

# get a test image
img_url = (
    "https://lucid-static.storage.googleapis.com/building-blocks/examples/dog_cat.png"
)
img_tf = load(img_url)
img_pt = torch.as_tensor(img_tf.transpose(2, 0, 1))[None, ...]
img_pil = Image.open(requests.get(img_url, stream=True).raw)

# instantiate ported model
net = googlenet(pretrained=True)

# get predictions
out = net(img_pt)
logits = out.detach().numpy()[0]
top_k = np.argsort(-logits)[:5]

# load labels
labels = load(InceptionV1.labels_path, split=True)

# show predictions
for i, k in enumerate(top_k):
    prediction = logits[k]
    label = labels[k]
    print(f"{i}: {label} ({prediction*100:.2f}%)")

# transforms


# def build_grid(source_size, target_size):
#     k = float(target_size) / float(source_size)
#     direct = (
#         torch.linspace(0, k, target_size)
#         .unsqueeze(0)
#         .repeat(target_size, 1)
#         .unsqueeze(-1)
#     )
#     full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
#     return full.cuda()


# def random_crop_grid(x, grid):
#     d = x.size(2) - grid.size(1)
#     grid = grid.repeat(x.size(0), 1, 1, 1).cuda()
#     # Add random shifts by x
#     grid[:, :, :, 0] += torch.FloatTensor(x.size(0)).cuda().random_(0, d).unsqueeze(
#         -1
#     ).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
#     # Add random shifts by y
#     grid[:, :, :, 1] += torch.FloatTensor(x.size(0)).cuda().random_(0, d).unsqueeze(
#         -1
#     ).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
#     return grid


# # We want to crop a 80x80 image randomly for our batch
# # Building central crop of 80 pixel size
# grid_source = build_grid(224, 80)
# # Make radom shift for each batch
# grid_shifted = random_crop_grid(batch, grid_source)
# # Sample using grid sample
# sampled_batch = F.grid_sample(batch, grid_shifted)


from clarity.pytorch.transform import RandomSpatialJitter, RandomUpsample

# crop = torchvision.transforms.RandomCrop(
#     224, padding=34, pad_if_needed=True, padding_mode="reflect"
# )
jitter = RandomSpatialJitter(16)
ups = RandomUpsample()
for i in range(10):
    cropped = ups(img_pt)
    show(cropped.numpy()[0].transpose(1, 2, 0))
    # display(cropped)


# result = param().cpu().detach().numpy()[0].transpose(1, 2, 0)
# loss_curve = objective.history

# 2019-11-21 notes from Pytorch team
# Set up model
# net = googlenet(pretrained=True)
# parameterization = Image()  # TODO: make size adjustable, currently hardcoded
# input_image = parameterization()

# writer = SummaryWriter()
# writer.add_graph(net, (input_image,))
# writer.close()

# Specify target module / "objective"
# target_module = net.mixed3b._pool_reduce[1]
# target_channel = 54
# hook = OutputHook(target_module) # TODO: investigate detach on rerun
# parameterization = Image()  # TODO: make size adjustable, currently hardcoded
# optimizer = optim.Adam(parameterization.parameters, lr=0.025)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = net.to(device)
# parameterization = parameterization.to(device)
# for i in range(1000):
#     optimizer.zero_grad()

#     # forward pass through entire net
#     input_image = parameterization()
#     with suppress(AbortForwardException):
#         _ = net(input_image.to(device))

#     # activations were stored during forward pass
#     assert hook.saved_output is not None
#     loss = -hook.saved_output[:, target_channel, :, :].sum()  # channel 13

#     loss.backward()
#     optimizer.step()

#     if i % 100 == 0:
#         print("Loss: ", -loss.cpu().detach().numpy())
#         url = show(
#             parameterization.raw_image.cpu()
#             .detach()
#             .numpy()[0]
#             .transpose(1, 2, 0)
#         )

# traced_net = torch.jit.trace(net, example_inputs=(input_image,))
# print(traced_net.graph)
