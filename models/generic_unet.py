# models/generic_unet.py
# 来自 nnUNet 框架的轻量通用 U-Net，用作 DDPM 中的 anchor 分支增强模块

import torch
import torch.nn as nn
from .nn import conv_nd, layer_norm

# 简化版 FFParser：频域调制增强分支
class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2) * 0.02)

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x.float(), dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        return x.view(B, C, H, W)


# Generic UNet: 用于 anchor 辅助增强模块
class Generic_UNet(nn.Module):
    def __init__(self, in_channels, base_num_features, out_channels, num_pool, anchor_out=True, upscale_logits=True):
        super().__init__()
        self.anchor_out = anchor_out
        self.upscale_logits = upscale_logits

        self.down1 = nn.Sequential(
            conv_nd(2, in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            conv_nd(2, 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            conv_nd(2, 32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            conv_nd(2, 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            conv_nd(2, 64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = conv_nd(2, 128, out_channels, 1)

    def forward(self, x, hs=None):
        # anchor encoder
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.bottleneck(x4)

        if self.anchor_out:
            return (x1, x3), self.final(self.up(x5))
        else:
            return x5, self.final(self.up(x5))
