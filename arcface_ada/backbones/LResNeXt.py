from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
# import torch.functional as F
import torch
from collections import namedtuple
import math
import pdb
from collections import OrderedDict
import torch.nn.functional as F

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class bottleneck_IR(Module): # LResNet100E-IR
    # https://zhuanlan.zhihu.com/p/139095264
    # https://www.ecohnoch.cn/2018/12/17/shuxue79/

    expansion = 1
    def __init__(self, in_channel, depth, stride, groups=1, base_width=64):
    # def __init__(self, in_channel, depth, stride, groups=2, base_width=40):
        super(bottleneck_IR, self).__init__()
        depth2 = int(in_channel * (base_width / 64.)) * groups
        print((depth, depth2))

        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        # else:
        #     self.shortcut_layer = Sequential(
        #         Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        # self.res_layer = Sequential(
        #     BatchNorm2d(in_channel),
        #     Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False),
        #     PReLU(depth),
        #     Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False),
        #     BatchNorm2d(depth))

        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            Conv2d(in_channel, depth2, 1, bias=False),
            BatchNorm2d(depth2),
            Conv2d(depth2, depth2, 3, stride, 1, 1, groups, bias=False),
            BatchNorm2d(depth2),
            Conv2d(depth, depth * self.expansion, 1, bias=False),
            BatchNorm2d(depth * self.expansion),
            PReLU(depth)
            )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        # return res + shortcut      
        return res + x   

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    """
    在這幾個函數中，blocks這個list首先定義其殘差結構，具體來說，就是每個殘差塊包含多少個殘差單元，以及stride等等一些東西，
    再通過resnet_v2這個函數將blocks展開，並且定義殘差網絡的圖模型
    """
    """
    ResNet-50  -> [3,4,6,3]
    ResNet-101 -> [3,4,23,3]
    ResNet-152 -> [3,8,36,3]
    ResNet-200  -> [3,24,36,3]
    """
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]    
    elif num_layers == 101:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=23),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        # [3, 8, 36, 3]
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    elif num_layers == 200:
        # [3, 24, 36, 3]
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=24),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 269:
        # [3, 30, 48, 8]
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            # get_block(in_channel=64, depth=128, num_units=30),
            # get_block(in_channel=128, depth=256, num_units=48),
            # get_block(in_channel=256, depth=512, num_units=8)
        ]
    elif num_layers == 1000:
        # custom
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=49),
            get_block(in_channel=128, depth=256, num_units=70),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

##################################  ResNet Backbone #############################################################

class Backbone(Module):
    def __init__(self, num_layers, mode='ir', dropout = 0.4, fp16=False, **kwargs):
        super(Backbone, self).__init__()
        assert num_layers in [18, 50, 101, 152, 200, 269], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.fp16 = fp16
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        
        
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(dropout),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512, affine=False))
        modules = []
        for block in blocks:
            print("---------------------------------------------")
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)


    def forward(self,x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.input_layer(x)
            x = self.body(x)
        x = self.output_layer(x.float() if self.fp16 else x) # torch.Size([14, 512, 7, 7])
        return x # torch.Size([8, 512])  


def resnet_18(mode='ir', **kwargs):
    return Backbone(18, mode=mode, **kwargs)

def resnet_50(mode='ir', **kwargs):
    return Backbone(50, mode=mode, **kwargs)

def resnet_101(mode='ir', **kwargs):
    # ================================================================
    # Total params: 54,202,304
    # Trainable params: 54,202,304
    # Non-trainable params: 0
    # ----------------------------------------------------------------
    # Input size (MB): 0.14
    # Forward/backward pass size (MB): 162.70
    # Params size (MB): 206.77
    # Estimated Total Size (MB): 369.61
    # ----------------------------------------------------------------
    return Backbone(101, mode=mode, **kwargs)

def resnet_152(mode='ir', **kwargs):
    # ================================================================
    # Total params: 70,736,576
    # Trainable params: 70,736,576
    # Non-trainable params: 0
    # ----------------------------------------------------------------
    # Input size (MB): 0.14
    # Forward/backward pass size (MB): 218.98
    # Params size (MB): 269.84
    # Estimated Total Size (MB): 488.96
    # ----------------------------------------------------------------

    return Backbone(152, mode=mode, **kwargs)

def resnet_200(mode='ir', **kwargs):
    return Backbone(200, mode=mode, **kwargs)

def resnet_269(mode='ir', **kwargs):
    return Backbone(269, mode=mode, **kwargs)

