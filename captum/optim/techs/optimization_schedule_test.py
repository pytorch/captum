# %load_ext autoreload
# %autoreload 2

from typing import Dict, Callable, Iterable

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from kornia.losses import total_variation

from lucid.misc.io import save
from clarity.pytorch.io import show
from clarity.pytorch.models import googlenet
from clarity.pytorch.param import (
    NaturalImage,
    TransformationRobustness,
    RandomAffine,
    GaussianSmoothing,
    BlendAlpha,
)
from clarity.pytorch.optim.objectives import (
    InputOptimization,
    single_target_objective,
    channel_activation,
    neuron_activation,
)
from clarity.pytorch.optim.optimize import optimize, n_steps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def alpha_neuron(target, input):
    def innr(mapping):
        acts = mapping[target]
        input_value = mapping[input]
        _, _, H, W = acts.shape
        obj = acts[:, 8, H // 2, W // 2]
        mean_alpha = input_value[3].mean()
        mean_tv = total_variation(input_value[3:])
        return obj * (1 - mean_alpha) - mean_tv

    return innr


def run():
    net = googlenet(pretrained=True).to(device)

    robustness_transforms = nn.Sequential(
        TransformationRobustness(jitter=True),
        TransformationRobustness(jitter=True),
        TransformationRobustness(jitter=True),
        # GaussianSmoothing(channels=3, kernel_size=(3, 3), sigma=1),
        # TransformationRobustness(jitter=True),
        RandomAffine(),
        BlendAlpha(),
    )

    param = NaturalImage((112, 112), channels=4, color_correct=True, normalize=False).to(
        device
    )

    target = net.mixed3a._1x1
    objective = InputOptimization(
        net=net,
        input_param=param,
        transform=robustness_transforms,
        targets=[target, param],
        loss_function=alpha_neuron(target, param),
    )

    optimize(objective, n_steps(128))
    result = objective.input_param()
    save(result.detach().numpy().transpose(1, 2, 0), "image.png")
    return result.cpu()


run()


def two_stag_opt():

    net = googlenet(pretrained=True).to(device)

    robustness_transforms = nn.Sequential(
        RandomAffine(rotate=True, scale=True, translate=True, shear=True), BlendAlpha()
    )

    param = NaturalImage((112, 112), channels=3).to(device)

    target = net.mixed3a._1x1
    objective = InputOptimization(
        net=net,
        input_param=param,
        transform=robustness_transforms,
        targets=[target, param],
        loss_function=alpha_neuron(target, param),
    )

    optimizer = optim.Adam(objective.parameters(), lr=0.025)

    history = []
    step = 0
    for step in tqdm(range(256)):
        optimizer.zero_grad()

        loss_value = objective.loss()
        history.append(loss_value.cpu().detach().numpy())
        (-1 * loss_value.mean()).backward()
        optimizer.step()
        step += 1

    return history
