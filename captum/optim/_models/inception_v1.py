from typing import Optional, Tuple, Union, cast

import torch
import torch.nn as nn

from captum.optim._utils.models import (
    AvgPool2dConstrained,
    CatLayer,
    Conv2dSame,
    LocalResponseNormLayer,
    RedirectedReluLayer,
    ReluLayer,
    SkipLayer,
)

GS_SAVED_WEIGHTS_URL = (
    "https://github.com/pytorch/captum/raw/"
    + "optim-wip/captum/optim/_models/inception5h.pth"
)


def googlenet(
    pretrained: bool = False,
    progress: bool = True,
    model_path: Optional[str] = None,
    **kwargs
):
    r"""GoogLeNet (also known as Inception v1 & Inception 5h) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to
            stderr
        model_path (str, optional): Optional path for InceptionV1 model file.
        replace_relus_with_redirectedrelu (bool, optional): If True, return pretrained
            model with Redirected ReLU in place of ReLU layers.
        use_linear_modules_only (bool, optional): If True, return pretrained
            model with all nonlinear layers replaced with linear equivalents.
        aux_logits (bool, optional): If True, adds two auxiliary branches that can
            improve training. Default: *False* when pretrained is True otherwise *True*
        out_features (int, optional): Number of output features in the model used for
            training. Default: 1008 when pretrained is True.
        transform_input (bool, optional): If True, preprocesses the input according to
            the method with which it was trained on ImageNet. Default: *False*
        bgr_transform (bool, optional): If True and transform_input is True, perform an
            RGB to BGR transform in the internal preprocessing.
            Default: *False*
    """

    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "bgr_transform" not in kwargs:
            kwargs["bgr_transform"] = False
        if "replace_relus_with_redirectedrelu" not in kwargs:
            kwargs["replace_relus_with_redirectedrelu"] = True
        if "use_linear_modules_only" not in kwargs:
            kwargs["use_linear_modules_only"] = False
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
        return model

    return InceptionV1(**kwargs)


# Better version of Inception V1 / GoogleNet for Inception5h
class InceptionV1(nn.Module):
    def __init__(
        self,
        out_features: int = 1008,
        aux_logits: bool = False,
        transform_input: bool = False,
        bgr_transform: bool = False,
        replace_relus_with_redirectedrelu: bool = False,
        use_linear_modules_only: bool = False,
    ) -> None:
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.bgr_transform = bgr_transform
        lrn_vals = (11, 0.0011, 0.5, 2.0)

        if use_linear_modules_only:
            activ = SkipLayer
            pool = AvgPool2dConstrained
        else:
            if replace_relus_with_redirectedrelu:
                activ = RedirectedReluLayer
            else:
                activ = ReluLayer
            pool = nn.MaxPool2d

        self.conv1 = Conv2dSame(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3,
            groups=1,
            bias=True,
        )
        self.conv1_relu = activ()
        self.pool1 = pool(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.local_response_norm1 = LocalResponseNormLayer(*lrn_vals)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv2_relu = activ()
        self.conv3 = Conv2dSame(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=1,
            bias=True,
        )
        self.conv3_relu = activ()
        self.local_response_norm2 = LocalResponseNormLayer(*lrn_vals)

        self.pool2 = pool(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.mixed3a = InceptionModule(192, 64, 96, 128, 16, 32, 32, activ, pool)
        self.mixed3b = InceptionModule(256, 128, 128, 192, 32, 96, 64, activ, pool)
        self.pool3 = pool(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.mixed4a = InceptionModule(480, 192, 96, 204, 16, 48, 64, activ, pool)

        if self.aux_logits:
            self.aux1 = AuxBranch(508, out_features, activ)

        self.mixed4b = InceptionModule(508, 160, 112, 224, 24, 64, 64, activ, pool)
        self.mixed4c = InceptionModule(512, 128, 128, 256, 24, 64, 64, activ, pool)
        self.mixed4d = InceptionModule(512, 112, 144, 288, 32, 64, 64, activ, pool)

        if self.aux_logits:
            self.aux2 = AuxBranch(528, out_features, activ)

        self.mixed4e = InceptionModule(528, 256, 160, 320, 32, 128, 128, activ, pool)
        self.pool4 = pool(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.mixed5a = InceptionModule(832, 256, 160, 320, 48, 128, 128, activ, pool)
        self.mixed5b = InceptionModule(832, 384, 192, 384, 48, 128, 128, activ, pool)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.4000000059604645)
        self.fc = nn.Linear(1024, out_features)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            assert x.dim() == 3 or x.dim() == 4
            assert x.min() >= 0.0 and x.max() <= 1.0
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = x * 255 - 117
            x = x[:, [2, 1, 0]] if self.bgr_transform else x
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self._transform_input(x)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.pool1(x)
        x = self.local_response_norm1(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x = self.conv3_relu(x)
        x = self.local_response_norm2(x)

        x = self.pool2(x)
        x = self.mixed3a(x)
        x = self.mixed3b(x)
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
        x = self.pool4(x)
        x = self.mixed5a(x)
        x = self.mixed5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        if not self.aux_logits:
            return cast(torch.Tensor, x)
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
        activ=ReluLayer,
        p_layer=nn.MaxPool2d,
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
        self.conv_1x1_relu = activ()

        self.conv_3x3_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c3x3reduce,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_3x3_reduce_relu = activ()
        self.conv_3x3 = Conv2dSame(
            in_channels=c3x3reduce,
            out_channels=c3x3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=1,
            bias=True,
        )
        self.conv_3x3_relu = activ()

        self.conv_5x5_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c5x5reduce,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_5x5_reduce_relu = activ()
        self.conv_5x5 = Conv2dSame(
            in_channels=c5x5reduce,
            out_channels=c5x5,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=1,
            groups=1,
            bias=True,
        )
        self.conv_5x5_relu = activ()

        self.pool = p_layer(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.pool_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=pool_proj,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.pool_proj_relu = activ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1x1 = self.conv_1x1(x)
        c1x1 = self.conv_1x1_relu(c1x1)

        c3x3 = self.conv_3x3_reduce(x)
        c3x3 = self.conv_3x3_reduce_relu(c3x3)
        c3x3 = self.conv_3x3(c3x3)
        c3x3 = self.conv_3x3_relu(c3x3)

        c5x5 = self.conv_5x5_reduce(x)
        c5x5 = self.conv_5x5_reduce_relu(c5x5)
        c5x5 = self.conv_5x5(c5x5)
        c5x5 = self.conv_5x5_relu(c5x5)

        px = self.pool(x)
        px = self.pool_proj(px)
        px = self.pool_proj_relu(px)
        return torch.cat([c1x1, c3x3, c5x5, px], dim=1)


class AuxBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 508,
        out_features: int = 1008,
        activ=ReluLayer,
    ) -> None:
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
        self.loss_conv_relu = activ()
        self.loss_fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.loss_fc_relu = activ()
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
