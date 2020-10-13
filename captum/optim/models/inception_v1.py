from __future__ import division

import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from .conv2d import Conv2dSame

from torch.hub import load_state_dict_from_url

# __all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

GS_SAVED_WEIGHTS_URL = (
    "https://storage.googleapis.com/openai-clarity/temp/InceptionV1_pytorch.pth"
)

GoogLeNetOutputs = namedtuple(
    "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
)
GoogLeNetOutputs.__annotations__ = {
    "logits": Tensor,
    "aux_logits2": Optional[Tensor],
    "aux_logits1": Optional[Tensor],
}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" not in kwargs:
            kwargs["aux_logits"] = False
        if kwargs["aux_logits"]:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, "
                "so make sure to train them"
            )
        original_aux_logits = kwargs["aux_logits"]
        kwargs["aux_logits"] = True
        kwargs["init_weights"] = False
        model = GoogLeNet(**kwargs)

        state_dict = load_state_dict_from_url(
            GS_SAVED_WEIGHTS_URL, progress=progress, check_hash=False
        )
        model.load_state_dict(state_dict)
        # if not original_aux_logits:
        #     model.aux_logits = False
        #     del model.aux1, model.aux2
        return model

    return GoogLeNet(**kwargs)


def _get_tf_value_by_name(name, graph, sess):
    op = graph.get_operation_by_name(name)
    return sess.run(op.values()[0])


def _import_weight_into_module(pt_param, tf_name, graph, sess):
    tf_value = _get_tf_value_by_name(tf_name, graph, sess)
    if len(tf_value.shape) == 4 and len(pt_param.shape) == 4:
        # assume k,k,c_in,c_out -> c_out,c_in,k,k
        tf_value_transposed = tf_value.transpose(3, 2, 0, 1)
        if tf_value_transposed.shape == pt_param.shape:
            pt_param.data = torch.as_tensor(tf_value_transposed)
        else:
            raise RuntimeError(
                f"non-matching shapes: {tf_value_transposed.shape} != {pt_param.shape}"
            )
    elif len(tf_value.shape) == 2 and len(pt_param.shape) == 2:
        if tf_value.shape == pt_param.shape:
            pt_param.data = torch.as_tensor(tf_value)
        elif tf_value.transpose(1, 0).shape == pt_param.shape:
            pt_param.data = torch.as_tensor(tf_value.transpose(1, 0))
        else:
            raise RuntimeError(
                f"non-matching shapes: {tf_value.shape} != {pt_param.shape}"
            )
    elif len(tf_value.shape) == 1 and len(pt_param.shape) == 1:
        if tf_value.shape == pt_param.shape:
            pt_param.data = torch.as_tensor(tf_value)
        else:
            raise RuntimeError(
                f"non-matching shapes: {tf_value.shape} != {pt_param.shape}"
            )
    else:
        raise NotImplementedError


def _tf_param_name_for_module(module, pt_param_name):
    if hasattr(module, "tf_param_name"):
        return module.tf_param_name(pt_param_name)

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        assert pt_param_name in ["weight", "bias"]
        return pt_param_name[0]  # will be w or b
    elif isinstance(module, nn.Sequential):
        sequence, pt_param_name = pt_param_name.split(".")
        assert pt_param_name in ["weight", "bias"]
        if int(sequence) == 0:
            return f"bottleneck_{pt_param_name[0]}"
        elif int(sequence) == 1 or int(sequence) == 2:
            return pt_param_name[0]
        else:
            raise NotImplementedError(f"cannot handle sequence blocks larger than 3")
    else:
        raise NotImplementedError(f"unknown module: {module}")


class GoogLeNet(nn.Module):
    # __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes=1008,
        aux_logits=True,
        transform_input=True,
        init_weights=True,
        blocks=None,
    ):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv2d0 = Conv2dSame(3, 64, kernel_size=7, stride=2, padding=3)
        # self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool0 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # nn.modules.LocalResponseNorm specifies size rather than radius
        tf_radius = 5
        pt_size = tf_radius * 2 + 1
        self.lrn = nn.LocalResponseNorm(pt_size, alpha=0.0001 * pt_size, beta=0.5, k=2)
        self.conv2d1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2d2 = Conv2dSame(64, 192, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.mixed3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.mixed3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.mixed4a = inception_block(480, 192, 96, 204, 16, 48, 64)
        self.mixed4b = inception_block(508, 160, 112, 224, 24, 64, 64)
        self.mixed4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.mixed4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.mixed4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool10 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.mixed5a = inception_block(832, 256, 160, 320, 48, 128, 128)
        self.mixed5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        # if aux_logits:
        #     self.aux1 = inception_aux_block(512, num_classes)
        #     self.aux2 = inception_aux_block(528, num_classes)

        self.avgpool0 = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.2)
        self.softmax2_pre_activation = nn.Linear(1024, num_classes)
        self.softmax2 = nn.Softmax()

        # if init_weights:
        #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             import scipy.stats as stats
    #             X = stats.truncnorm(-2, 2, scale=0.01)
    #             values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
    #             values = values.view(m.weight.size())
    #             with torch.no_grad():
    #                 m.weight.copy_(values)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            # x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            # x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            # x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            # x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            assert x.min() >= 0.0 and x.max() <= 1.0
            x = x * 255 - 117
        return x

    def _forward(self, x):
        # assert x.size(1) == 3
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv2d0(x)
        x = F.relu(x, inplace=True)
        # N x 64 x 112 x 112
        x = self.maxpool0(x)
        x = self.lrn(x)
        # N x 64 x 56 x 56
        x = self.conv2d1(x)
        x = F.relu(x, inplace=True)
        # N x 64 x 56 x 56
        x = self.conv2d2(x)
        x = F.relu(x, inplace=True)
        x = self.lrn(x)
        # N x 192 x 56 x 56
        x = self.maxpool1(x)

        # # N x 192 x 28 x 28
        # x = self.mixed3a(x)
        x = self.mixed3a(x)
        # # N x 256 x 28 x 28
        x = self.mixed3b(x)
        # # N x 480 x 28 x 28
        x = self.maxpool4(x)
        # # N x 480 x 14 x 14
        x = self.mixed4a(x)
        # # N x 512 x 14 x 14
        # aux_defined = self.training and self.aux_logits
        # if aux_defined:
        #     aux1 = self.aux1(x)
        # else:
        #     aux1 = None

        x = self.mixed4b(x)
        # # N x 512 x 14 x 14
        x = self.mixed4c(x)
        # # N x 512 x 14 x 14
        x = self.mixed4d(x)
        # # N x 528 x 14 x 14
        # if aux_defined:
        #     aux2 = self.aux2(x)
        # else:
        #     aux2 = None

        x = self.mixed4e(x)
        # # N x 832 x 14 x 14
        x = self.maxpool10(x)
        # # N x 832 x 7 x 7
        x = self.mixed5a(x)
        # # N x 832 x 7 x 7
        x = self.mixed5b(x)
        # # N x 1024 x 7 x 7

        x = self.avgpool0(x)
        # # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # # N x 1024
        # x = self.dropout(x)
        x = self.softmax2_pre_activation(x)
        x = self.softmax2(x)
        # # N x 1000 (num_classes)
        aux2, aux1 = None, None
        return x, aux2, aux1

    # @torch.jit.unused
    # def eager_outputs(self, x, aux2, aux1):
    #     # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
    #     if self.training and self.aux_logits:
    #         return _GoogLeNetOutputs(x, aux2, aux1)
    #     else:
    #         return x

    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        return x
        # aux_defined = self.training and self.aux_logits
        # if torch.jit.is_scripting():
        #     if not aux_defined:
        #         warnings.warn(
        #             "Scripted GoogleNet always returns GoogleNetOutputs Tuple"
        #         )
        #     return GoogLeNetOutputs(x, aux2, aux1)
        # else:
        #     return self.eager_outputs(x, aux2, aux1)

    def import_weights_from_tf(self, model):
        import tensorflow as tf

        print("Setting Paramaters…")
        with tf.Graph().as_default() as graph, tf.Session() as sess:
            tf.import_graph_def(model.graph_def)

            prefix = "import/"
            for module_name, module in self.named_children():
                print("named child", module_name)
                if module_name == "softmax2_pre_activation":
                    module_name = "softmax2"
                if not hasattr(module, "import_weights_from_tf"):
                    for param_name, pt_param in module.named_parameters(recurse=False):
                        tf_param_name = _tf_param_name_for_module(module, param_name)
                        tf_param_name = f"{prefix}{module_name}_{tf_param_name}"

                        print(
                            f"Setting {module_name}.{param_name} to value of {tf_param_name} ({pt_param.shape})"
                        )
                        _import_weight_into_module(pt_param, tf_param_name, graph, sess)
                else:
                    module.import_weights_from_tf(prefix, module_name, graph, sess)

        #     print(name, type(module))
        #     for param_name, pt_param in module.named_parameters(recurse=False):
        #         print('module param', param_name)

        # for name, pt_param in self.named_parameters(recurse=False):
        #     print('non-recurse', name)

        # for name, pt_param in self.named_parameters(recurse=True):
        #     print('recurse', name)

        # print("Setting Paramaters…")
        # with tf.Graph().as_default() as graph, tf.Session() as sess:
        #     tf.import_graph_def(model.graph_def)
        #     for name, pt_param in self.named_parameters(recurse=True):
        #         module_name, w_or_b = name.rsplit(".", 1)
        #         tf_name = f"import/{module_name}_{w_or_b[0]}"
        #         tf_name = tf_name.replace('.', '')
        #         _import_weight_into_module(pt_param, name, tf_name, graph, sess)


class Inception(nn.Module):
    # __constants__ = ["branch2", "branch3", "branch4"]
    # (192, 64, 96, 128, 16, 32, 32)
    def __init__(
        self,
        in_channels,  # 192
        ch1x1,  # 64
        ch3x3bottleneck,  # 96
        ch3x3,  # 128
        ch5x5bottleneck,  # 16
        ch5x5,  # 32
        pool_proj,  # 32
        conv_block=None,
    ):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = Conv2dSame
        self._1x1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self._3x3 = nn.Sequential(
            conv_block(in_channels, ch3x3bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_block(ch3x3bottleneck, ch3x3, kernel_size=3, padding=1),
        )

        self._5x5 = nn.Sequential(
            conv_block(in_channels, ch5x5bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_block(ch5x5bottleneck, ch5x5, kernel_size=5, padding=1),
        )

        self._pool_reduce = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x):
        _1x1 = self._1x1(x)
        # _3x3_bottleneck = self._3x3[0](x)
        _3x3 = self._3x3(x)
        _5x5 = self._5x5(x)
        # _5x5_bottleneck = self._5x5[0](x)
        _pool_reduce = self._pool_reduce(x)

        outputs = [_1x1, _3x3, _5x5, _pool_reduce]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return F.relu(torch.cat(outputs, 1), inplace=True)

    def import_weights_from_tf(self, prefix, own_name, graph, sess):
        for module_name, module in self.named_children():
            print(f"{own_name}: named child {module_name}")
            if not hasattr(module, "import_weights_from_tf"):
                for param_name, pt_param in module.named_parameters(recurse=True):
                    tf_param_name = _tf_param_name_for_module(module, param_name)
                    tf_param_name = f"{prefix}{own_name}{module_name}_{tf_param_name}"

                    print(
                        f"Setting {module_name}.{param_name} to value of {tf_param_name} ({pt_param.shape})"
                    )
                    _import_weight_into_module(pt_param, tf_param_name, graph, sess)
            else:
                module.import_weights_from_tf(module_name, prefix, graph, sess)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
