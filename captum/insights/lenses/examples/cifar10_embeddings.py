#!/usr/bin/env python3
from lenses import mbx
import os
import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from pathlib import Path
import os

this_filepath = Path(os.path.abspath(__file__))
this_dirpath = this_filepath.parent

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


dataset_dirname = "cifar10.dataset.log"
img_dataset = torchvision.datasets.CIFAR10(
    root=dataset_dirname, train=False, download=True
)


label_names = (
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
)


class Sample(mbx.sample.GenericSample):
    def __init__(self, image, label_name):
        super().__init__()
        self.label_name = label_name
        large_image = image.resize((128, 128))
        self.thumbnail = image.resize((32, 32))
        self.attr_request_handlers.enable_image_from_memory("image", large_image)

    def to_thumbnail(self):
        return self.thumbnail

    def to_payload(self):
        return {"label": self.label_name, "image": None}


def get_sample(sample_id):
    img, label_id = img_dataset[sample_id]
    label_name = label_names[label_id]
    return Sample(img, label_name)


def get_explorer(workspace, pretrained, clean, load_module_outputs=True):
    if pretrained:
        suffix = "pretrained"
    else:
        suffix = "random"
    cache_dirname = f"cifar10_{suffix}.explorer.cache.log"
    explorer_name = f"cifar10 {suffix}"

    dataset = torchvision.datasets.CIFAR10(
        root=dataset_dirname, train=False, download=True, transform=T.ToTensor()
    )

    if os.path.exists(cache_dirname) and not clean:
        print(f"loading cache... ({cache_dirname})")
        explorer = workspace.load_explorer(
            cache_dirname, explorer_name, get_sample, load_module_outputs
        )
    else:
        explorer = workspace.add_explorer(explorer_name, get_sample)
        model = Model()
        if pretrained:
            captum_dirpath = this_dirpath.parent.parent.parent.parent
            sd_filename = captum_dirpath.joinpath(
                "tutorials", "models", "cifar_torchvision.pt"
            )
            model.load_state_dict(torch.load(sd_filename))
        for name, module in model.named_children():
            explorer.add_module(name, module)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        def forward():
            for minibatch in tqdm(data_loader):
                x, _ = minibatch
                model(x)

        explorer.run(forward, 5)
        print(f"saving cache... ({cache_dirname})")
        explorer.save(cache_dirname, True)
    return explorer


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--batch-size", type=int, default=32)
arg_parser.add_argument("--num-workers", type=int, default=2)
arg_parser.add_argument("--clean", action="store_true")
arg_parser.add_argument("--dev-frontend-host", type=str, default=None)
args = arg_parser.parse_args()

workspace_cache_dirname = "cifar10.workspace.cache.log"

if os.path.exists(workspace_cache_dirname) and not args.clean:
    print(f"loading cache... ({workspace_cache_dirname})")
    workspace = mbx.Workspace.load(
        workspace_cache_dirname, get_sample=get_sample, load_module_outputs=False
    )
else:
    workspace = mbx.Workspace()
    # comparison between a pretrained model and a model with random weights
    explorer_pretrained = get_explorer(workspace, pretrained=True, clean=args.clean)
    explorer_random = get_explorer(workspace, pretrained=False, clean=args.clean)
    workspace.compute_correlations()
    print(f"saving cache... ({workspace_cache_dirname})")
    workspace.save(workspace_cache_dirname, overwrite=True, save_module_outputs=True)

workspace.show(dev_frontend_host=args.dev_frontend_host)
