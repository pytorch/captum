import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import os
from flask import Flask, render_template

from torchvision import models

from captum import IntegratedGradients
import base64
from captum import DeepLift
from captum import NoiseTunnel
from captum import gradients
from captum import visualization as viz
from io import BytesIO
from PIL import Image
from api import AttributionVisualizer

import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)


def get_classes():
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    return classes


def get_pretrained_model():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    print("Using existing trained model")
    net.load_state_dict(torch.load("../../notebooks/models/cifar_torchvision.pt"))
    return net


dataiter = iter(testloader)
images, labels = dataiter.next()

outputs = net(images)
_, predicted = torch.max(outputs, 1)


def convert_img_base64(img, denormalize=False):
    if denormalize:
        img = img / 2 + 0.5

    buff = BytesIO()

    plt.imsave(buff, img)
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


def calculate_gradient(index):
    input = images[index].unsqueeze(0)
    input.requires_grad = True
    net.eval()

    ig = IntegratedGradients(net)
    net.zero_grad()

    attr_ig, delta = ig.attribute(
        input, baselines=input * 0, target=labels[index], n_steps=200
    )
    attr_ig = np.transpose(attr_ig[0].squeeze().cpu().detach().numpy(), (1, 2, 0))

    img_integrated_gradient_overlay = viz.visualize_image(
        attr_ig,
        np.transpose(images[index].cpu().detach().numpy(), (1, 2, 0)),
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=True,
        mask_mode=True,
    )
    ig_64 = convert_img_base64(img_integrated_gradient_overlay)
    img = np.transpose(images[index].cpu().detach().numpy(), (1, 2, 0))
    print(img.shape)
    img_64 = convert_img_base64(img, True)

    return ig_64, img_64


# plt.imsave("original.png", images[ind])
