from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from detectron2.modeling import Backbone
from typing import Callable, Optional
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
import torch.utils.checkpoint as cp
import timm


def _assert_strides_are_log2_contiguous(strides):
    """验证步长是否连续2的幂次"""
    for i in range(1, len(strides)):
        assert strides[i] == 2 * strides[i-1] or strides[i-1] == 2 * strides[i], \
            f"步长不连续: {strides}"

import math
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, ShapeSpec
from detectron2.layers import Conv2d, get_norm

import math
import torch
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer
from detectron2.modeling import Backbone, ShapeSpec

class TimmVITWrapper(Backbone):
    """支持任意输入尺寸的ViT Backbone封装"""
    def __init__(self, 
                 model_name='vit_small_patch16_224.augreg_in1k', 
                 pretrained=True,
                 out_feature="last_feat"):
        super().__init__()
        

        self.vit = timm.create_model(model_name, pretrained=pretrained)  
        

        self.patch_size = self._get_patch_size()
        self.embed_dim = self._get_embed_dim()

        if hasattr(self.vit, 'pos_embed'):
            self.register_buffer('original_pos_embed', self.vit.pos_embed.data)
            self.vit.pos_embed = None  
            

        if hasattr(self.vit.patch_embed, 'strict_img_size'):
            self.vit.patch_embed.strict_img_size = False
            

        self._out_feature = out_feature
        self._out_feature_channels = {out_feature: self.embed_dim}
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_features = [out_feature]

    def _get_patch_size(self):
        """动态获取patch尺寸"""
        if hasattr(self.vit.patch_embed, 'patch_size'):
            return self.vit.patch_embed.patch_size[0]
        elif hasattr(self.vit.patch_embed.proj, 'kernel_size'):
            return self.vit.patch_embed.proj.kernel_size[0]
        else:
            raise ValueError("无法自动获取patch_size")

    def _get_embed_dim(self):
        """动态获取嵌入维度"""
        if hasattr(self.vit, 'pos_embed'):
            return self.vit.pos_embed.shape[-1]
        elif len(self.vit.blocks) > 0:
            return self.vit.blocks[0].mlp.fc1.in_features
        else:
            raise ValueError("无法自动获取embed_dim")

    def _interpolate_pos_encoding(self, x, H_patch, W_patch):
        """动态位置编码插值核心方法"""
        if not hasattr(self, 'original_pos_embed'):
            return 0  
        
        cls_pos = self.original_pos_embed[:, :1]  # [1, 1, D]
        patch_pos = self.original_pos_embed[:, 1:]  # [1, N, D]
        
        orig_H = int(math.sqrt(patch_pos.shape[1]))
        orig_W = orig_H
        
        patch_pos = patch_pos.reshape(1, orig_H, orig_W, -1).permute(0, 3, 1, 2)  # [1, D, H, W]
        
        patch_pos = F.interpolate(
            patch_pos,
            size=(H_patch, W_patch),
            mode='bicubic',
            align_corners=False
        )
        
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H_patch * W_patch, -1)  # [1, N, D]
        return torch.cat([cls_pos, patch_pos], dim=1).to(x.device)

    def forward(self, x):
        """处理任意尺寸的完整流程"""
        B, C, H, W = x.shape
        
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = self.vit.patch_embed(x)  # [B, N, D]
        
        if hasattr(self.vit, 'cls_token'):
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)  # [B, N+1, D]
        
        H_patch = (H + pad_h) // self.patch_size
        W_patch = (W + pad_w) // self.patch_size
        if hasattr(self, 'original_pos_embed'):
            pos_embed = self._interpolate_pos_encoding(x, H_patch, W_patch)
            x += pos_embed
        
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        
        if hasattr(self.vit, 'cls_token'):
            features = x[:, 1:]  # 移除CLS token [B, N, D]
        else:
            features = x
        
        features = features.permute(0, 2, 1).reshape(B, -1, H_patch, W_patch)
        
        return {self._out_feature: features}

    def output_shape(self):
        return {
            self._out_feature: ShapeSpec(
                channels=self.embed_dim,
                stride=self.patch_size
            )
        }

class SimpleFeaturePyramid(Backbone):
    """增强版特征金字塔，支持动态尺寸"""
    def __init__(self, 
                 ViT,
                 in_feature='last_feat',  
                 out_channels=256,
                 scale_factors=[4.0, 2.0, 1.0, 0.5],
                 norm="LN",
                 square_pad=0):
        super().__init__()
        self.in_feature = in_feature
        self.ViT = ViT
        
        input_shapes = ViT.output_shape()
        self.strides = [int(input_shapes[in_feature].stride / s) for s in scale_factors]
        _assert_strides_are_log2_contiguous(self.strides)
        
        dim = input_shapes[in_feature].channels
        self.stages = nn.ModuleList()
        
        for scale in scale_factors:
            layers = []
            if scale == 4.0:
                layers += [
                    nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2),
                    get_norm(norm, dim//2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim//2, dim//4, kernel_size=2, stride=2),
                ]
                out_dim = dim//4
            elif scale == 2.0:
                layers += [nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2)]
                out_dim = dim//2
            elif scale == 1.0:
                layers += [nn.Identity()]
                out_dim = dim
            elif scale == 0.5:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                out_dim = dim
            
            layers += [
                Conv2d(out_dim, out_channels, 1, 
                      norm=get_norm(norm, out_channels),
                      padding_mode='reflect'),
                Conv2d(out_channels, out_channels, 3, 
                      padding=1, 
                      norm=get_norm(norm, out_channels),
                      padding_mode='reflect')
            ]
            self.stages.append(nn.Sequential(*layers))
        
        self._out_feature_strides = {
            f"res{int(math.log2(s))}": s for s in self.strides
        }
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}

    def forward(self, x):

        features = self.ViT(x)[self.in_feature]
        
        B, C, H_patch, W_patch = features.shape

        outputs = []
        for stage in self.stages:
            out = stage(features)
            outputs.append(out)
        
        return {k: v for k, v in zip(self._out_features, outputs)}

@BACKBONE_REGISTRY.register()
class VITCOMMON1(SimpleFeaturePyramid, Backbone):
    """增强版Backbone实现，支持动态输入"""
    def __init__(self, cfg, input_shape):
        vit_cfg = cfg.MODEL.VITCOMMON1
        
        vit_backbone = TimmVITWrapper(
            model_name=vit_cfg.MODEL_NAME,
            pretrained=vit_cfg.PRETRAINED,
            out_feature=vit_cfg.IN_FEATURE
        )
        
        super().__init__(
            ViT=vit_backbone,
            in_feature=vit_cfg.IN_FEATURE,
            out_channels=vit_cfg.OUT_CHANNELS,
            scale_factors=vit_cfg.SCALE_FACTORS,
            norm=vit_cfg.NORM,
        )
        self._out_features = vit_cfg.OUT_FEATURES
        base_stride = vit_backbone.output_shape()[vit_cfg.IN_FEATURE].stride
        
        self._out_feature_strides = {
            f"res{int(math.log2(int(base_stride/s)))}": int(base_stride/s)
            for s in vit_cfg.SCALE_FACTORS
        }
        self._out_feature_channels = {
            k: vit_cfg.OUT_CHANNELS for k in self._out_features
        }

    @property
    def size_divisibility(self):
        return self.ViT.patch_size

    def forward(self, x):
        outputs = super().forward(x)
        
        for k in outputs:
            B, C, H, W = outputs[k].shape
            outputs[k] = outputs[k][:, :, :H, :W]
            
        return {k: v for k, v in outputs.items() if k in self._out_features}