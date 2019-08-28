from typing import List, Dict, Callable, Any, Optional
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from serve import start_server
import numpy as np
from captum import IntegratedGradients
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from captum import visualization as viz
import inspect
from collections import namedtuple

VisualizationOutput = namedtuple(
    "VisualizationOutput", "feature_output actual predicted"
)
FeatureOutput = namedtuple("FeatureOutput", "base modified type")


class BaseFeature:
    def __init__(self, name: str):
        self.name = name

    def visualization_type(self):
        raise NotImplementedError


class ImageFeature(BaseFeature):
    def __init__(self, name: str):
        super().__init__(name)

    def visualization_type(self):
        return "image"

    def visualize(self, attribution, data, label):
        data_t = np.transpose(data.cpu().detach().numpy(), (1, 2, 0))
        attribution_t = np.transpose(
            attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )

        img_integrated_gradient_overlay = viz.visualize_image(
            attribution_t,
            data_t,
            clip_above_percentile=99,
            clip_below_percentile=0,
            overlay=True,
            mask_mode=True,
        )
        ig_64 = convert_img_base64(img_integrated_gradient_overlay)
        img_64 = convert_img_base64(data_t, True)

        return FeatureOutput(
            base=img_64, modified=ig_64, type=self.visualization_type()
        )


class TextFeature(BaseFeature):
    def __init__(self, name: str):
        super().__init__(name)


class ComplexFeature(BaseFeature):
    def __init__(self, name: str):
        super().__init__(name)


# sparse, dense, etc


# class VisualizerCaller(object):
#     def __init__(self, visualizer, dataset, transform):
#         self.visualizer = visualizer
#         self.dataset = dataset
#         self.transform = transform


#     def visualize(self):
#         # start flask server

#         start_server()


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


def convert_img_base64(img, denormalize=False):
    if denormalize:
        img = img / 2 + 0.5

    buff = BytesIO()

    plt.imsave(buff, img)
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


class AttributionVisualizer(object):
    def __init__(
        self, models: Any, classes: List[str], features: List[BaseFeature], dataset: Any
    ):
        self.classes = classes
        self.features = features
        self.models = models
        self.dataset = dataset

    def _calculate_attribution(self, net, data, label):
        input = data.unsqueeze(0)
        input.requires_grad = True
        net.eval()
        ig = IntegratedGradients(net)
        net.zero_grad()
        attr_ig, _ = ig.attribute(input, baselines=input * 0, target=label)
        return attr_ig[0]  # why the first one?

    def render(self):
        return start_server(self)

    def visualize(self, n):
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=10, shuffle=False, num_workers=2
        )
        images, labels = iter(loader).next()
        net = self.models[0]
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        outputs = []
        for i in range(n):
            image, label = images[i], labels[i]
            attribution = self._calculate_attribution(net, image, label)
            for feature in self.features:
                output = feature.visualize(attribution, image, label)
            actual_label = self.classes[labels[i]]
            predicted_label = self.classes[predicted[i].item()]
            outputs.append(
                VisualizationOutput(
                    feature_output=output,
                    actual=actual_label,
                    predicted=predicted_label,
                )
            )
        print(outputs)
        return outputs


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    visualizer = AttributionVisualizer(
        models=[get_pretrained_model()],  # some nn.Module
        classes=get_classes(),  # a list of classes, indices correspond to name. If scalar or one output, just print out whatever
        features=[ImageFeature("Image")],  # output visualization type
        dataset=dataset,  # should also support regular iter
    )

    visualizer.render()


def example_api_please_ignore():
    visualizer = AttributionVisualizer(
        models=[model, model1],  # if list, else single model
        # index_to_label=["banana", "apple"],
        features=[
            ComplexFeature("Video", start=500),
            BaseFeature("Video", start=0),
            ComplexFeature("Features", classes=["height", "width"]),
            ImageFeature("Photo"),
            TextFeature(
                "Comment", itos=lambda x: x, stoi=lambda x: x
            ),  # optional, for direct target
            TextFeature("Title", convert_to_text=embedding_to_text),
            BaseFeature("Sparse features"),
        ],
    )

    transform = lambda x: x

    dataset = ["some", "items"]  # can accept additional args
    single = "one item"

    dataset_text = [(data_tensor, index_tensor)]
    # multiple datasets
    widget = visualizer(
        [dataset, dataset2], transform=transform
    )  # on each batch, before input to model

    # single dataset
    widget = visualizer(dataset, transform=transform)

    # single item
    widget = visualizer(single, transform=None)

    # approx_method="riemann_left",
    # config={'approx_method': 'riemann'}

    # meet with multimo
