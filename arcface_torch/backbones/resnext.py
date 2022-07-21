from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
# import torch.functional as F
import torch
from collections import namedtuple
import math
import pdb
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn

##################################  ResNeXt Backbone #############################################################
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        # torch.nn.functional.adaptive_avg_pool2d(input, output_size)[SOURCE]
        # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        # self.relu = nn.ReLU(True)
        self.relu = PReLU(width)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class ResNext(nn.Module):
    fc_scale = 7 * 7
    num_features = 512
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, dilated=False, norm_layer=nn.BatchNorm2d, dropout = 0.4, fp16=False, **kwargs):
        super(ResNext, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.fp16 = fp16


        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                         bias=False),
        #     norm_layer(self.inplanes),
        #     nn.ReLU(inplace=True)
        # )     
        # 
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     norm_layer(self.inplanes),
        #     PReLU(self.inplanes),
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #     norm_layer(self.inplanes),
        #     PReLU(self.inplanes),
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),

        #     norm_layer(self.inplanes),
        #     PReLU(self.inplanes),
        # )          

        self.conv1 = nn.Sequential(                                                                                                       
                nn.Conv2d(3, self.inplanes//2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.inplanes//2),
                PReLU(self.inplanes//2),
                nn.Conv2d(self.inplanes//2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                PReLU(self.inplanes),     
            )

        # self.conv1 = nn.Sequential(                                                                                                       
        #         nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #         norm_layer(self.inplanes),
        #         PReLU(self.inplanes),
        #         nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        #         norm_layer(self.inplanes),
        #         PReLU(self.inplanes),     
        #     )

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        self.out1 = Sequential(
                                BatchNorm2d(512*block.expansion),
                                Dropout(dropout),
                                Flatten(),                      
                                ) 
        self.fc = nn.Linear(self.num_features * block.expansion * self.fc_scale, self.num_features)
        self.features = nn.BatchNorm1d(self.num_features, affine=False)


        # -----------------------------------------------------------------------
        # self.out1 = Sequential(
        #                         GlobalAvgPool2d(), # GAP
        #                         BatchNorm1d(512*block.expansion),
        #                         Dropout(dropout),
        #                         Flatten(),                      
        #                         ) 
        # self.fc = nn.Linear(self.num_features * block.expansion, self.num_features)
        # self.features = nn.BatchNorm1d(self.num_features, eps=1e-05)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, dilation=2, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x) # torch.Size([14, 2048, 7, 7])

            x = self.out1(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        return x

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

"""
    ResNeXt
    https://github.com/Hsuxu/ResNeXt/blob/3680fa351615727c52fce0edd35d91b7f171076d/models.py#L123
    <cardinality, bottleneck_width> =>    (1, 64). (2, 40), (4, 24), (8, 14), (32, 4)
    <groups, width_per_group> =>    (1, 64). (2, 40), (4, 24), (8, 14), (32, 4)
"""

def resnext50_32x4d(pretrained=False, **kwargs):
    num_blocks = get_num_block(50)
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext100_32x4d(pretrained=False, **kwargs):
    num_blocks = get_num_block(100)
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext101_32x4d(pretrained=False, **kwargs):
    num_blocks = get_num_block(101)
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
  
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext101_8x14d(pretrained=False, **kwargs):
    num_blocks = get_num_block(101)
    kwargs['groups'] = 8
    kwargs['width_per_group'] = 14
   
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext101_4x24d(pretrained=False, **kwargs):
    num_blocks = get_num_block(101)
    kwargs['groups'] = 4
    kwargs['width_per_group'] = 24
   
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext101_2x40d(pretrained=False, **kwargs):
    num_blocks = get_num_block(101)
    kwargs['groups'] = 2
    kwargs['width_per_group'] = 40
  
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext101_1x64d(pretrained=False, **kwargs):
    num_blocks = get_num_block(101)
    kwargs['groups'] = 1
    kwargs['width_per_group'] = 64
   
    return ResNext(Bottleneck, num_blocks, **kwargs)    

def resnext152_32x4d(pretrained=False, **kwargs):
    num_blocks = get_num_block(152)
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    
    return ResNext(Bottleneck, num_blocks, **kwargs)


def resnext152_8x14d(pretrained=False, **kwargs):
    num_blocks = get_num_block(152)
    kwargs['groups'] = 8
    kwargs['width_per_group'] = 14
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext152_4x24d(pretrained=False, **kwargs):
    num_blocks = get_num_block(152)
    kwargs['groups'] = 4
    kwargs['width_per_group'] = 24
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext152_2x40d(pretrained=False, **kwargs):
    num_blocks = get_num_block(152)
    kwargs['groups'] = 2
    kwargs['width_per_group'] = 40
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext152_1x64d(pretrained=False, **kwargs):
    num_blocks = get_num_block(152)
    kwargs['groups'] = 1
    kwargs['width_per_group'] = 64
    return ResNext(Bottleneck, num_blocks, **kwargs)


def resnext200_32x4d(pretrained=False, **kwargs):
    num_blocks = get_num_block(200)
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext200_8x14d(pretrained=False, **kwargs):
    num_blocks = get_num_block(200)
    kwargs['groups'] = 8
    kwargs['width_per_group'] = 14
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext200_4x24d(pretrained=False, **kwargs):
    num_blocks = get_num_block(200)
    kwargs['groups'] = 4
    kwargs['width_per_group'] = 24
    return ResNext(Bottleneck, num_blocks, **kwargs)    

def resnext200_2x40d(pretrained=False, **kwargs):
    num_blocks = get_num_block(200)
    kwargs['groups'] = 2
    kwargs['width_per_group'] = 40
    return ResNext(Bottleneck, num_blocks, **kwargs) 
    

def resnext200_1x64d(pretrained=False, **kwargs):
    num_blocks = get_num_block(200)
    kwargs['groups'] = 1
    kwargs['width_per_group'] = 64
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext269_32x4d(pretrained=False, **kwargs):
    num_blocks = get_num_block(269)
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNext(Bottleneck, num_blocks, **kwargs)

def resnext269_4x24d(pretrained=False, **kwargs):
    num_blocks = get_num_block(269)
    kwargs['groups'] = 4
    kwargs['width_per_group'] = 24
    return ResNext(Bottleneck, num_blocks, **kwargs)  

    

