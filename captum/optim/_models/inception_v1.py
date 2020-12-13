import torch
import torch.nn as nn
import torch.nn.functional as F

import captum.optim._utils.models as model_utils

GS_SAVED_WEIGHTS_URL = (
    "https://github.com/pytorch/captum/raw/"
    + "optim-wip/captum/optim/_models/inception5h.pth"
)


def googlenet(
    pretrained: bool = False, progress: bool = True, model_path: str = None, **kwargs
):
    r"""GoogLeNet (also known as Inception v1 & Inception 5h) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        model_path (str): Optional path for InceptionV1 model file
        aux_logits (bool): If True, adds two auxiliary branches that can improve
            training. Default: *False* when pretrained is True otherwise *True*
        out_features (int): Number of output features in the model used for
            training. Default: 1008 when pretrained is True.
        transform_input (bool): If True, preprocesses the input according to
            the method with which it was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" not in kwargs:
            kwargs["aux_logits"] = False
        if "out_features" not in kwargs:
            kwargs["out_features"] = 1008

        model = InceptionV1(**kwargs)

        if model_path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                GS_SAVED_WEIGHTS_URL, progress=progress, check_hash=False
            )
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model_utils.replace_layers(model)
        return model

    return InceptionV1(**kwargs)


# Better version of Inception V1/GoogleNet for Inception5h
class InceptionV1(nn.Module):
    def __init__(
        self,
        out_features: int = 1008,
        aux_logits: bool = False,
        transform_input: bool = False,
    ) -> None:
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        lrn_vals = (9, 9.99999974738e-05, 0.5, 1.0)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            groups=1,
            bias=True,
        )
        self.conv1_relu = model_utils.ReluLayer()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.localresponsenorm1 = model_utils.LocalResponseNormLayer(*lrn_vals)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv2_relu = model_utils.ReluLayer()
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv3_relu = model_utils.ReluLayer()
        self.localresponsenorm2 = model_utils.LocalResponseNormLayer(*lrn_vals)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.mixed3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.mixed3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.mixed4a = InceptionModule(480, 192, 96, 204, 16, 48, 64)

        if self.aux_logits:
            self.aux1 = AuxBranch(508, out_features)

        self.mixed4b = InceptionModule(508, 160, 112, 224, 24, 64, 64)
        self.mixed4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.mixed4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)

        if self.aux_logits:
            self.aux2 = AuxBranch(528, out_features)

        self.mixed4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.mixed5a = InceptionModule(832, 256, 160, 320, 48, 128, 128)
        self.mixed5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.4000000059604645)
        self.fc = nn.Linear(1024, out_features)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            assert x.dim() == 3 or x.dim() == 4
            assert x.min() >= 0.0 and x.max() <= 1.0
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = x * 255 - 117
            x = x.clone()[:, [2, 1, 0]]  # RGB to BGR
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._transform_input(x)
        x = F.pad(x, (2, 3, 2, 3))
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = F.pad(x, (0, 1, 0, 1), value=float("-inf"))
        x = self.pool1(x)
        x = self.localresponsenorm1(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = F.pad(x, (1, 1, 1, 1)) 
        x = self.conv3(x)
        x = self.conv3_relu(x)
        x = self.localresponsenorm2(x)

        x = F.pad(x, (0, 1, 0, 1), value=float("-inf"))
        x = self.pool2(x)
        x = self.mixed3a(x)
        x = self.mixed3b(x)
        x = F.pad(x, (0, 1, 0, 1), value=float("-inf"))
        x = self.pool3(x)
        x = self.mixed4a(x)

        if self.aux_logits:
            aux1_output = self.aux1(x)

        x = self.mixed4b(x)
        x = self.mixed4c(x)
        x = self.mixed4d(x)

        if self.aux_logits:
            aux2_output = self.aux2(x)

        x = self.mixed4e(x)
        x = F.pad(x, (0, 1, 0, 1), value=float("-inf"))
        x = self.pool4(x)
        x = self.mixed5a(x)
        x = self.mixed5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        if not self.aux_logits:
            return x
        else:
            return x, aux1_output, aux2_output


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        c1x1: int,
        c3x3reduce,
        c3x3: int,
        c5x5reduce: int,
        c5x5: int,
        pool_proj: int,
    ) -> None:
        super(InceptionModule, self).__init__()
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c1x1,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_1x1_relu = model_utils.ReluLayer()

        self.conv_3x3_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c3x3reduce,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_3x3_reduce_relu = model_utils.ReluLayer()
        self.conv_3x3 = nn.Conv2d(
            in_channels=c3x3reduce,
            out_channels=c3x3,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_3x3_relu = model_utils.ReluLayer()

        self.conv_5x5_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c5x5reduce,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_5x5_reduce_relu = model_utils.ReluLayer()
        self.conv_5x5 = nn.Conv2d(
            in_channels=c5x5reduce,
            out_channels=c5x5,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_5x5_relu = model_utils.ReluLayer()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.pool_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=pool_proj,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pool_proj_relu = model_utils.ReluLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1x1 = self.conv_1x1(x)
        c1x1 = self.conv_1x1_relu(c1x1)

        c3x3 = self.conv_3x3_reduce(x)
        c3x3 = self.conv_3x3_reduce_relu(c3x3)
        c3x3 = F.pad(c3x3, (1, 1, 1, 1))  
        c3x3 = self.conv_3x3(c3x3)      
        c3x3 = self.conv_3x3_relu(c3x3)

        c5x5 = self.conv_5x5_reduce(x)
        c5x5 = self.conv_5x5_reduce_relu(c5x5)
        c5x5 = F.pad(c5x5, (2, 2, 2, 2))
        c5x5 = self.conv_5x5(c5x5)
        c5x5 = self.conv_5x5_relu(c5x5)

        px = self.pool_proj(x)
        px = self.pool_proj_relu(px)
        px = F.pad(px, (1, 1, 1, 1), value=float("-inf"))
        px = self.pool(px)
        return torch.cat([c1x1, c3x3, c5x5, px], dim=1)


class AuxBranch(nn.Module):
    def __init__(self, in_channels: int = 508, out_features: int = 1008) -> None:
        super(AuxBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.loss_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.loss_conv_relu = model_utils.ReluLayer()
        self.loss_fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.loss_fc_relu = model_utils.ReluLayer()
        self.loss_dropout = nn.Dropout(0.699999988079071)
        self.loss_classifier = nn.Linear(
            in_features=1024, out_features=out_features, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.loss_conv(x)
        x = self.loss_conv_relu(x)
        x = torch.flatten(x, 1)
        x = self.loss_fc(x)
        x = self.loss_fc_relu(x)
        x = self.loss_dropout(x)
        x = self.loss_classifier(x)
        return x
