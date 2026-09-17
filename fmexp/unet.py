"""小型 3D U-Net 主干（方案 §3.1）。

A 与 B 使用完全相同的卷积块、下采样和上采样方式，只有输入/输出通道数不同。
四级通道数 [16, 32, 64, 128]，GroupNorm（适合小 batch）。
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int, groups: int) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            _gn(cout, groups),
            nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            _gn(cout, groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int = 16, channel_mult: Sequence[int] = (1, 2, 4, 8),
                 norm_groups: int = 8):
        super().__init__()
        chs = [base_channels * m for m in channel_mult]
        self.chs = chs
        self.stem = ConvBlock(in_channels, chs[0], norm_groups)

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.downs.append(ConvBlock(chs[i], chs[i + 1], norm_groups))
            self.pools.append(nn.MaxPool3d(2))

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose3d(chs[i], chs[i - 1], 2, stride=2))
            self.decs.append(ConvBlock(chs[i - 1] * 2, chs[i - 1], norm_groups))

        self.head = nn.Conv3d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: Tuple[torch.Tensor, ...] = ()
        x = self.stem(x)
        skips = skips + (x,)
        for down, pool in zip(self.downs, self.pools):
            x = pool(x)
            x = down(x)
            skips = skips + (x,)
        # skips: [level0, level1, level2, level3]
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(method: str, cfg) -> nn.Module:
    """method: 'A' 直接分割；'B' flow matching。"""
    if method == "A":
        return UNet3D(4 + 4, cfg.n_classes, cfg.base_channels, cfg.channel_mult, cfg.norm_groups)
    if method == "B":
        # 4 MRI + 4 存在标记 + K noisy mask + 1 时间 t
        return UNet3D(4 + 4 + cfg.n_classes + 1, cfg.n_classes,
                      cfg.base_channels, cfg.channel_mult, cfg.norm_groups)
    raise ValueError(method)
