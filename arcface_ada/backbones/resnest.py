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


class SplitAttention(nn.Module):
    '''
    split attention class
    '''
    def __init__(self,
        in_channels,
        channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        radix=2,
        reduction_factor=4
    ):
        super(SplitAttention, self).__init__()

        self.radix = radix

        self.radix_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels*radix,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups*radix,
                bias=bias
            ),
            nn.BatchNorm2d(channels*radix),
            nn.ReLU(inplace=True)
            # PReLU(channels*radix),
        )   

        self.pool = nn.AvgPool2d(3)  

        inter_channels = max(32, in_channels*radix//reduction_factor)

        self.attention = nn.Sequential( #fc1
            nn.Conv2d(
                in_channels=channels,
                out_channels=inter_channels,
                kernel_size=1,
                groups=groups
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            # PReLU(inter_channels),
            nn.Conv2d( #fc2
                in_channels=inter_channels,
                out_channels=channels*radix,
                kernel_size=1,
                groups=groups
            )
        )

        self.rsoftmax = rSoftMax(
            groups=groups,
            radix=radix
        )

        

    def forward(self, x):
        
        # NOTE: comments are ugly...

        '''
        input  : |             in_channels               |
        '''

        '''
        radix_conv : |                radix 0            |               radix 1             | ... |                radix r            |
                     | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k | ... | group 0 | group 1 | ... | group k |
        '''
        x = self.radix_conv(x)

        '''
        split :  [ | group 0 | group 1 | ... | group k |,  | group 0 | group 1 | ... | group k |, ... ]
        sum   :  | group 0 | group 1 | ...| group k |
        '''
        B, rC = x.size()[:2]
        splits = torch.split(x, rC // self.radix, dim=1)
        gap = sum(splits)

        # gap = F.adaptive_avg_pool2d(gap, 1) # => ncnn segment fault 
        # gap = self.pool(gap) 
        gap = F.avg_pool2d(gap, 3) 

        '''
        !! becomes cardinal-major !!
        attention : |             group 0              |             group 1              | ... |              group k             |
                    | radix 0 | radix 1| ... | radix r | radix 0 | radix 1| ... | radix r | ... | radix 0 | radix 1| ... | radix r |
        '''
        att_map = self.attention(gap)

        '''
        !! transposed to radix-major in rSoftMax !!
        rsoftmax : same as radix_conv
        '''
        att_map = self.rsoftmax(att_map)

        '''
        split : same as split
        sum : same as sum
        '''
        att_maps = torch.split(att_map, rC // self.radix, dim=1)
        out = sum([att_map*split for att_map, split in zip(att_maps, splits)])


        '''
        output : | group 0 | group 1 | ...| group k |
        concatenated tensors of all groups,
        which split attention is applied
        '''

        return out.contiguous()

class rSoftMax(nn.Module):
    '''
    (radix-majorize) softmax class
    input is cardinal-major shaped tensor.
    transpose to radix-major
    '''
    def __init__(self,
        groups=1,
        radix=2
    ):
        super(rSoftMax, self).__init__()

        self.groups = groups
        self.radix = radix

    def forward(self, x):
        B = x.size(0)
        # transpose to radix-major
        x = x.view(B, self.groups, self.radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(B, -1, 1, 1)

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
            self.conv2 = SplitAttention(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, 
                reduction_factor=4)  # Forward/backward pass size (MB): 599.21, 112,538,744  693.32         
                # reduction_factor=32)  # Forward/backward pass size (MB): 596.16, 110,899,208
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        # self.out3 = nn.Sequential(
        #     nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False),
        #     norm_layer(planes*4)            
        # )

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
        # out = self.out3(out)
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
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False,
                 dropblock_prob=0.0,
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



# <groups, width_per_group> =>    (1, 64). (2, 40), (4, 24), (8, 14), (32, 4)
def resnest101_8x14d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(101)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=8, bottleneck_width=14,
                   deep_stem=False, stem_width=64, avg_down=False,
                   avd=False, avd_first=False, **kwargs)   
    return model

def resnest152_8x14d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=8, bottleneck_width=14,
                   deep_stem=False, stem_width=64, avg_down=False,
                   avd=False, avd_first=False, **kwargs)   
    return model

def resnest152_32x4d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=32, bottleneck_width=4,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest200_8x14d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=8, bottleneck_width=14,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest200_2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=2, bottleneck_width=40,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model


def resnest200_4x24d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=4, bottleneck_width=24,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest152_1x64d_r4(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(152)
    model = ResNet(Bottleneck, num_blocks,
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model   

def resnest200_1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model

def resnest200_1x64d_r4(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(200)
    model = ResNet(Bottleneck, num_blocks,
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model 

def resnest269_1x64d_r4(pretrained=False, root='~/.encoding/models', **kwargs):
    num_blocks = get_num_block(256)
    model = ResNet(Bottleneck, num_blocks,
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=False, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)   
    return model 



# Ref2:
#     https://github.com/MachineLP/Pytorch_multi_task_classifier/blob/fce381cc51759c91513f2c7c20f010d537f1d993/qdnet_classifier/models/resnest.py
#     https://github.com/MachineLP/Pytorch_multi_task_classifier/blob/fce381cc51759c91513f2c7c20f010d537f1d993/qdnet_classifier/models/helpers.py
#     https://github.com/MachineLP/Pytorch_multi_task_classifier/blob/fce381cc51759c91513f2c7c20f010d537f1d993/qdnet_classifier/models/resnet.py

