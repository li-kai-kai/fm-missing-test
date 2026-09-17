"""训练损失（方案 §4）。

A: 前景 soft Dice + 交叉熵，权重均为 1，不叠加类别权重。
B: 纯速度 MSE。第一轮不加入 SDF / 区域加权 / 辅助 Dice。
两者的 loss 数值含义不同，不可横向比较大小。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target_fg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """logits [B,2,...]；target_fg [B,...] in {0,1}。Dice 用 softmax 后的肿瘤概率。"""
    prob_fg = torch.softmax(logits, dim=1)[:, 1]
    dims = tuple(range(1, prob_fg.ndim))
    num = 2 * (prob_fg * target_fg).sum(dim=dims) + eps
    den = prob_fg.sum(dim=dims) + target_fg.sum(dim=dims) + eps
    return 1.0 - (num / den).mean()


def ce_loss(logits: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
    """交叉熵直接吃 logits。target_cls [B,...] in {0,1}。"""
    return F.cross_entropy(logits, target_cls.long())


def loss_A(logits: torch.Tensor, y1: torch.Tensor) -> tuple:
    if y1.shape[1] == 2:
        d = soft_dice_loss(logits, y1[:, 1])
    else:
        # 每例各非背景互斥类别等权，先算 soft Dice 再平均；CE 覆盖全部四类。
        probs, target = torch.softmax(logits, dim=1)[:, 1:], y1[:, 1:]
        dims = tuple(range(2, probs.ndim))
        d = 1 - ((2 * (probs * target).sum(dims) + 1e-6) /
                 (probs.sum(dims) + target.sum(dims) + 1e-6)).mean()
    c = ce_loss(logits, y1.argmax(dim=1))
    return d + c, {"dice": d.detach(), "ce": c.detach()}


def loss_B(pred_v: torch.Tensor, target_v: torch.Tensor) -> tuple:
    mse = F.mse_loss(pred_v, target_v)
    return mse, {"mse": mse.detach()}
