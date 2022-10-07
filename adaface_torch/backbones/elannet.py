import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as T
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter


model_urls = {
    "elannet": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet.pth",
}

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='silu',             # activation
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=False))
            convs.append(nn.BatchNorm2d(c1))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=s, padding=0, dilation=d, groups=1, bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, model_size='large', act_type='silu', depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        if model_size == 'large':
            depth = 2
        elif model_size == 'tiny':
            depth = 1
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, depthwise=depthwise)
            for _ in range(depth)
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, depthwise=depthwise)
            for _ in range(depth)
        ])

        self.out = Conv(inter_dim*4, out_dim, k=1)



    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out


class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='silu'):
        super().__init__()
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out

# https://github.com/yjh0410/image_classification_pytorch/blob/c730925e5a690a33133560616fdde97d65066b7c/models/elannet.py#L215
# ELANNet of YOLOv7
class ELANNet(nn.Module):
    """
    ELAN-Net of YOLOv7.
    """
    fc_scale = 7 * 7
    def __init__(self, depthwise=False, model_size='large', num_features=1000, dropout= 0.4, fp16=False):
        super(ELANNet, self).__init__()
        if model_size == 'large':
            act_type = 'silu'
            final_dim = 1024
        elif model_size == 'tiny':
            act_type = 'lrelu'
            final_dim = 512
        
        if model_size == 'large':
            # large backbone
            self.layer_1 = nn.Sequential(
                Conv(3, 32, k=3, p=1, act_type='silu', depthwise=depthwise),      
                Conv(32, 64, k=3, p=1, s=2, act_type='silu', depthwise=depthwise),
                Conv(64, 64, k=3, p=1, act_type='silu', depthwise=depthwise)                                                   # P1/2
            )
            self.layer_2 = nn.Sequential(   
                Conv(64, 128, k=3, p=1, s=2, act_type='silu', depthwise=depthwise),             
                ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5, act_type='silu', depthwise=depthwise)                     # P2/4
            )
            self.layer_3 = nn.Sequential(
                DownSample(in_dim=256, act_type='silu'),             
                ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5, act_type='silu', depthwise=depthwise)                     # P3/8
            )
            self.layer_4 = nn.Sequential(
                DownSample(in_dim=512, act_type='silu'),             
                ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5, act_type='silu', depthwise=depthwise)                    # P4/16
            )
            self.layer_5 = nn.Sequential(
                DownSample(in_dim=1024, act_type='silu'),             
                ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25, act_type='silu', depthwise=depthwise)                  # P5/32
            )    

        elif model_size == 'tiny':
            # tiny backbone
            self.layer_1 = Conv(3, 32, k=3, p=1, s=2, act_type='lrelu', depthwise=depthwise)                                   # P1/2

            self.layer_2 = nn.Sequential(   
                Conv(32, 64, k=3, p=1, s=2, act_type='lrelu', depthwise=depthwise),             
                ELANBlock(in_dim=64, out_dim=64, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                              # P2/4
            )
            self.layer_3 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),             
                ELANBlock(in_dim=64, out_dim=128, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                              # P3/8
            )
            self.layer_4 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),             
                ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                              # P4/16
            )
            self.layer_5 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),             
                ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5,
                        model_size='tiny', act_type='lrelu', depthwise=depthwise)                                               # P5/32
            )

        self.fp16 = fp16
        # define transformt o resize the image with given size
        self.transform = T.Resize(size = (112*2,112*2))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(final_dim* self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False  

        self.output_layer = Sequential(BatchNorm2d(1024),
                                       Dropout(dropout),
                                       Flatten(),
                                       Linear(1024 * 7 * 7, 512),
                                       BatchNorm1d(512, affine=False)
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.transform(x)
            x = self.layer_1(x)
            x = self.layer_2(x)
            x = self.layer_3(x)
            x = self.layer_4(x)  
            x = self.layer_5(x) # torch.Size([1, 1024, 7, 7])
            # x = self.layer_6(x) # torch.Size([1, 1024, 7, 7])

            # [B, C, H, W] -> [B, C, 1, 1]
            # x = self.avgpool(x)
            # x = F.avg_pool2d(x, 4) 
            # [B, C, 1, 1] -> [B, C]
            # x = x.flatten(1)
        # x = self.linear(x.float() if self.fp16 else x)
        # x = self.features(x)
        x = self.output_layer(x.float() if self.fp16 else x) # torch.Size([14, 512, 7, 7])        
        return x

def build_elannet(pretrained=False, model_size='large', **kwargs):
    # model
    model = ELANNet(model_size=model_size, **kwargs)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['elannet']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model