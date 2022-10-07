from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200, iresnet400
from .mobilefacenet import get_mbf
from .resnet import resnet_18, resnet_50, \
                              resnet_101, resnet_152, \
                              resnet_200, resnet_269    

# from .LResNeXt import resnet_18, resnet_50, \
#                               resnet_101, resnet_152, \
#                               resnet_200, resnet_269                

from .resnext import resnext200_32x4d, resnext200_1x64d, resnext200_8x14d, \
                            resnext200_4x24d, resnext200_2x40d, resnext269_4x24d, \
                            resnext101_1x64d, resnext101_2x40d, resnext101_4x24d, resnext101_8x14d, resnext101_32x4d, \
                            resnext152_32x4d, resnext152_8x14d, resnext152_4x24d, resnext152_2x40d, resnext152_1x64d

from .resnest import resnest152_8x14d, resnest101_8x14d, \
    resnest200_8x14d, resnest200_2x40d, resnest200_4x24d, resnest200_1x64d, \
    resnest152_1x64d_r4, resnest200_1x64d_r4

from .elannet import build_elannet   
from .repvggnet import RepVGG_A0, RepVGG_B3, RepVGG_B2g4, RepVGG_B3g4

def get_model(name, **kwargs):
    # LResNetE-IR
    if name == "resnet_269":
        return resnet_269(mode='ir', **kwargs)
    elif name == "resnet_200":
        return resnet_200(mode='ir', **kwargs)
    
    
    # resnext (focus 8x14d)
    elif name == "resnext200_1x64d":
        return resnext200_1x64d(**kwargs)
    elif name == "resnext200_2x40d":
        return resnext200_2x40d(**kwargs)
    elif name == "resnext200_4x24d":
        return resnext200_4x24d(**kwargs)
    elif name == "resnext200_8x14d":
        return resnext200_8x14d(**kwargs)
    elif name == "resnext200_32x4d":
        return resnext200_32x4d(**kwargs)
    elif name == "resnext152_8x14d":
        return resnext152_8x14d(**kwargs)
    elif name == "resnext101_8x14d":
        return resnext101_8x14d(**kwargs)


    # resnest (focus 8x14d) (radix=2)
    elif name == "resnest101_8x14d":
        return resnest101_8x14d(**kwargs)
    elif name == "resnest152_8x14d":
        return resnest152_8x14d(**kwargs)
  
    # RepVgg
    elif name == "RepVGG_A0":
        return RepVGG_A0(**kwargs)
    elif name == "RepVGG_B3g4":
        return RepVGG_B3g4(**kwargs)
    elif name == "RepVGG_B3":
        return RepVGG_B3(**kwargs)


    elif name == "elannet_tiny":
        return build_elannet(model_size = "tiny", **kwargs)
    elif name == "elannet":
        return build_elannet(**kwargs)
        

    # resnet
    elif name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r400":
        return iresnet400(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    else:
        raise ValueError()
