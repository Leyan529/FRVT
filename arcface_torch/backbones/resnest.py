from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
# import torch.functional as F
import torch
from collections import namedtuple
import math
import pdb
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair

# from mxnet.context import cpu
# from mxnet.gluon.block import HybridBlock
# from mxnet.gluon import nn
# from mxnet.gluon.nn import BatchNorm
# import mxnet as mx
# from mxnet.gluon.nn import Conv2D, Block, HybridBlock, Dense, BatchNorm, Activation

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# class DropBlock2D(object):
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
       https://zhuanlan.zhihu.com/p/425636663
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                            groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        # self.relu = ReLU(inplace=True)
        self.relu1 = PReLU(channels*radix)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.relu2 = PReLU(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu1(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu2(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        # torch.nn.functional.adaptive_avg_pool2d(input, output_size)[SOURCE]
        # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False) # split k
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, 
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob) 
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)

        self.relu = PReLU(group_width)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out        

class ResNet(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    num_features = 512
    fc_scale = 7 * 7
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 dilated=False, dilation=1,
                 deep_stem=True, stem_width=64, avg_down=False,
                 avd=False, avd_first=False,
                 dropblock_prob=0.4,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, dropout = 0.4, fp16=False, **kwargs):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.fp16 = fp16

        super(ResNet, self).__init__()

        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                PReLU(stem_width),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                PReLU(stem_width),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),

                norm_layer(stem_width*2),
                PReLU(stem_width*2),
            )
        else:
            self.conv1 = nn.Sequential(                                                                                                       
                conv_layer(3, stem_width//2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width//2),
                PReLU(stem_width//2),
                conv_layer(stem_width//2, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                PReLU(stem_width),     
            )    


        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)

        self.out1 = Sequential(
                                BatchNorm2d(512*block.expansion),
                                Dropout(dropout),
                                Flatten(),                      
                                ) 
        self.fc = nn.Linear(self.num_features * block.expansion * self.fc_scale, self.num_features)
        self.features = nn.BatchNorm1d(self.num_features, affine=False)


        
        # ------------------------------------------------------------------------
        # self.out1 = Sequential(
        #                         GlobalAvgPool2d(),
        #                         BatchNorm1d(512*block.expansion),
        #                         Dropout(dropout),
        #                         Flatten(),                      
        #                         ) 
        # self.fc = nn.Linear(self.num_features * block.expansion, self.num_features)
        # self.features = nn.BatchNorm1d(self.num_features, eps=1e-05)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, 
                                # rectified_conv=self.rectified_conv,
                                # rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, 
                                # rectified_conv=self.rectified_conv,
                                # rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, 
                                # rectified_conv=self.rectified_conv,
                                # rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)         
          
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x) # torch.Size([2, 2048, 4, 4])


            # x = self.a(x)
            x = self.out1(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        return x

"""
    ResNeSt
    https://github.com/zhanghang1989/ResNeSt/blob/43c1c4b5c91898c75a50fd70867f32c6cd9aeef5/resnest/torch/models/resnest.py#L30
    https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/resnest.py
    <groups, bottleneck_width> =>    (1, 64). (2, 40), (4, 24), (8, 14), (32, 4)
    radix => divide parts
"""
def get_num_block(num_layers):
    """
    ResNet18、ResNet34: expansion=1
    ResNet50、ResNet101、ResNet152: expansion=4
    """
    num_blocks = []

    if num_layers == 50:
        num_blocks = [3, 4, 14, 3]
    elif num_layers == 100:
        num_blocks = [3, 13, 30, 3]
    elif num_layers == 101:
        num_blocks = [3, 4, 23, 3]
    elif num_layers == 152:
        num_blocks = [3, 8, 36, 3]
    elif num_layers == 200:
        num_blocks = [3, 24, 36, 3] 
    elif num_layers == 269:
        num_blocks = [3, 30, 48, 8] 

    return num_blocks

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(50)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(101)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def resnest152(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(269)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model

# <groups, width_per_group> =>    (1, 64). (2, 40), (4, 24), (8, 14), (32, 4)
def resnest152_1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest152_2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest152_4x24d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=4, bottleneck_width=24,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest152_8x14d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=8, bottleneck_width=14,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest152_32x4d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=32, bottleneck_width=4,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest200_8x14d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=8, bottleneck_width=14,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest200_2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model


def resnest200_4x24d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=4, bottleneck_width=24,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest152_1x64d_r4(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model   

def resnest200_1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest200_1x64d_r4(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model 

def resnest269_1x64d_r4(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(256)
    model = ResNet(Bottleneck, num_blocks,
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model 