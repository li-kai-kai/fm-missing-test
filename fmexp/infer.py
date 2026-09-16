"""推理（方案 §5）。

A: 滑窗预测 -> 融合连续分类分数 -> 最后统一 argmax。不先对 patch 二值化。
B: 从与完整体积同形状的噪声出发，做 N 次 Euler 更新；
   每一步所有滑窗都从**同一份当前 y** 读取，融合出完整速度场后统一更新 y。
   —— 同一步内不得边预测窗口边改 y，否则窗口顺序会改变结果。
   —— 重叠区域共享同一份初始噪声与当前状态，不为每个窗口重采噪声。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .data import case_noise_key


# ------------------------------------------------------------ 滑窗几何

def window_starts(size: int, window: int, stride: int) -> List[int]:
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)  # 末窗贴边，保证覆盖
    return starts


def plan_windows(shape: Sequence[int], window: int, overlap: float) -> Tuple[List[Tuple[slice, ...]], torch.Tensor]:
    """返回窗口切片列表与对应的加权权重体积 [1,1,w,w,w]。"""
    stride = max(1, int(round(window * (1.0 - overlap))))
    zs = window_starts(shape[0], window, stride)
    ys = window_starts(shape[1], window, stride)
    xs = window_starts(shape[2], window, stride)
    slices = [(slice(z, z + window), slice(y, y + window), slice(x, x + window))
              for z in zs for y in ys for x in xs]

    # 可分离的 Hann 窗（带下限），中心权重高，减轻块缝
    n = torch.arange(window, dtype=torch.float32)
    w1 = 0.5 - 0.5 * torch.cos(2 * torch.pi * (n + 0.5) / window)
    w1 = w1.clamp_min(1e-2)
    w3 = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).unsqueeze(0).unsqueeze(0)
    return slices, w3


# ------------------------------------------------------------ 输入拼装

def _expand_avail(avail: torch.Tensor, shape) -> torch.Tensor:
    """存在标记 [4] -> [4,D,H,W] 空间常量通道。"""
    return avail.view(4, 1, 1, 1).expand(4, *shape)


def build_input_A(mri: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """A 输入 = 4 MRI + 4 存在标记 = 8 通道。"""
    return torch.cat([mri, _expand_avail(avail, mri.shape[-3:])], dim=0)


def build_input_B(yt: torch.Tensor, t: float, mri: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """B 输入 = 4 MRI + 4 存在标记 + 2 noisy mask + 1 时间常量通道 = 11 通道。"""
    shp = mri.shape[-3:]
    tc = torch.full((1, *shp), float(t), dtype=mri.dtype, device=mri.device)
    return torch.cat([mri, _expand_avail(avail, shp), yt, tc], dim=0)


# ------------------------------------------------------------ A: 滑窗分类

@torch.no_grad()
def predict_volume_A(model, mri: torch.Tensor, avail: torch.Tensor, cfg: Config,
                     chunk: int = 32) -> np.ndarray:
    """返回前景概率体积 [D,H,W] float32。融合连续分数后再取类别。"""
    model.eval()
    dev = mri.device
    cond = build_input_A(mri, avail)                      # [8,D,H,W]
    slices, wgt = plan_windows(cond.shape[-3:], cfg.sw_window, cfg.sw_overlap)
    wgt = wgt.to(dev)

    acc = torch.zeros((2, *cond.shape[-3:]), dtype=torch.float32, device=dev)
    wsum = torch.zeros((1, *cond.shape[-3:]), dtype=torch.float32, device=dev)

    ins = torch.empty((len(slices),) + tuple(cond.shape[:-3]) + (cfg.sw_window,) * 3,
                      dtype=torch.float32, device=dev)
    for i, sl in enumerate(slices):
        ins[i] = cond[(slice(None),) + sl]

    for s in range(0, len(slices), chunk):
        x = ins[s:s + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
            logits = model(x.float())
        probs = torch.softmax(logits.float(), dim=1)
        w = wgt.expand(probs.shape[0], -1, -1, -1, -1)
        for j, sl in enumerate(slices[s:s + chunk]):
            acc[(slice(None),) + sl] += probs[j] * w[j]
            wsum[(slice(None),) + sl] += w[j]

    prob_fg = (acc[1] / wsum[0].clamp_min(1e-6))
    return prob_fg.cpu().numpy()


# ------------------------------------------------------------ B: 全体积 Euler

@torch.no_grad()
def _velocity_field(model, y: torch.Tensor, t: float, cond_win: torch.Tensor,
                    slices, wgt: torch.Tensor, shape, cfg: Config, chunk: int = 32) -> torch.Tensor:
    """一步：所有窗口读取同一份当前 y，融合出完整速度场。"""
    dev = y.device
    n = len(slices)
    yw = torch.empty((n, 2, cfg.sw_window, cfg.sw_window, cfg.sw_window),
                     dtype=torch.float32, device=dev)
    for i, sl in enumerate(slices):
        yw[i] = y[(slice(None),) + sl]

    wsum = torch.zeros((1, *shape), dtype=torch.float32, device=dev)
    out = torch.zeros((2, *shape), dtype=torch.float32, device=dev)

    for s in range(0, n, chunk):
        yb = yw[s:s + chunk]
        cb = cond_win[s:s + chunk]
        tc = torch.full((yb.shape[0], 1, cfg.sw_window, cfg.sw_window, cfg.sw_window),
                        float(t), dtype=torch.float32, device=dev)
        x = torch.cat([cb, yb, tc], dim=1)                # [k, 11, w,w,w]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
            v = model(x)
        v = v.float()
        w = wgt.expand(v.shape[0], -1, -1, -1, -1)
        for j, sl in enumerate(slices[s:s + chunk]):
            out[(slice(None),) + sl] += v[j] * w[j]
            wsum[(slice(None),) + sl] += w[j]

    return out / wsum.clamp_min(1e-6)


@torch.no_grad()
def predict_volume_B(model, mri: torch.Tensor, avail: torch.Tensor, cfg: Config,
                     init_noise: torch.Tensor, steps: int | None = None,
                     chunk: int = 32, collect_states: bool = False) -> np.ndarray:
    """从噪声积分生成分割。返回离散预测 [D,H,W] uint8（或含中间状态的 dict）。"""
    model.eval()
    dev = mri.device
    steps = steps or cfg.fm_steps
    shape = tuple(mri.shape[-3:])
    slices, wgt = plan_windows(shape, cfg.sw_window, cfg.sw_overlap)
    wgt = wgt.to(dev)

    cond = build_input_A(mri, avail)                      # MRI + 存在标记，全程固定
    cond_win = torch.empty((len(slices), 8, cfg.sw_window, cfg.sw_window, cfg.sw_window),
                           dtype=torch.float32, device=dev)
    for i, sl in enumerate(slices):
        cond_win[i] = cond[(slice(None),) + sl]

    y = init_noise.to(dev).float().clone()                # [2,D,H,W]，三场景共用同一份
    dt = 1.0 / steps
    states = {}
    if collect_states:
        # 中间状态记录**连续**前景通道 y[1]−y[0]，不是 argmax。
        # 中途 argmax 会把几乎是噪声的状态说成“全脑都是肿瘤”，那是错的读法：
        # 只有全部更新结束后 y≈y1，argmax 才是分割。
        states[0.0] = (y[1] - y[0]).cpu().numpy().astype(np.float32)

    for k in range(steps):
        t = k / steps
        v = _velocity_field(model, y, t, cond_win, slices, wgt, shape, cfg, chunk=chunk)
        y = y + dt * v                                    # 连续状态，不做裁剪、不中途 argmax
        if collect_states:
            frac = (k + 1) / steps
            if any(abs(frac - s) < 1e-9 for s in (0.25, 0.5, 0.75, 1.0)):
                states[frac] = (y[1] - y[0]).cpu().numpy().astype(np.float32)

    pred = y.argmax(dim=0).cpu().numpy().astype(np.uint8)  # 全部更新结束后才离散化
    if collect_states:
        return {"pred": pred, "states": states}
    return pred


def init_noise_for_case(cid: str, shape: Sequence[int], eval_seed: int, device) -> torch.Tensor:
    """按“病例 ID + 采样种子”固定初始噪声；同一病例三场景共用（方案 §5.2）。"""
    g = torch.Generator(device="cpu").manual_seed(case_noise_key(cid, eval_seed))
    return torch.randn((2,) + tuple(shape), generator=g, dtype=torch.float32).to(device)
