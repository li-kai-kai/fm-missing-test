"""评价指标与空 mask 处理规则（方案 §7.1）。

规则（预先约定，写入报告）：
  * 预测为空、真实非空  -> Dice = 0, 召回 = 0, HD95 记为 NaN 并单独计入“HD95 失败数”；
                            病例计入“完全漏检数”。绝不悄悄删除这类病例。
  * 两者均为空          -> Dice = 1（预先约定），召回无定义（NaN），HD95 无定义，
                            病例数单独列出。
  * 真实为空、预测非空  -> Dice = 0（本实验的测试病例均有肿瘤，不会出现）。
  * HD95 按真实体素间距（1mm 等体素）换算为毫米。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULT_SPACING = (1.0, 1.0, 1.0)


def dice_foreground(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    sp, sg = int(p.sum()), int(g.sum())
    if sp == 0 and sg == 0:
        return 1.0
    if sp == 0 or sg == 0:
        return 0.0
    return float(2.0 * np.logical_and(p, g).sum() / (sp + sg))


def recall_foreground(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    sg = int(g.sum())
    if sg == 0:
        return float("nan")
    return float(np.logical_and(p, g).sum() / sg)


def _surface(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    eroded = ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1),
                                    border_value=0)
    return m & ~eroded


def gt_hd95_cache(gt: np.ndarray, spacing=DEFAULT_SPACING) -> dict:
    """真实 mask 一侧的距离变换。

    它只取决于病例，与场景、模型、预测无关，因此在同一病例的多次评价中复用，
    可省掉一半的距离变换（全脑 240x240x155 的 EDT 是评价里最贵的一步）。
    """
    g = gt > 0
    if not g.any():
        return {"sg": None, "dt_to_g": None}
    sg = _surface(g)
    return {"sg": sg, "dt_to_g": ndimage.distance_transform_edt(~sg, sampling=spacing)}


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=DEFAULT_SPACING, gt_cache: dict | None = None) -> float:
    """95% Hausdorff 距离（毫米）。任一为空则返回 NaN。"""
    p, g = pred > 0, gt > 0
    if not p.any() or not g.any():
        return float("nan")
    cache = gt_cache if gt_cache is not None else gt_hd95_cache(g, spacing)
    sg, dt_to_g = cache["sg"], cache["dt_to_g"]
    sp = _surface(p)
    dt_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    all_d = np.concatenate([dt_to_g[sp], dt_to_p[sg]])
    return float(np.percentile(all_d, 95))


def case_metrics(pred: np.ndarray, gt: np.ndarray, spacing=DEFAULT_SPACING,
                 gt_cache: dict | None = None, fast: bool = False) -> dict:
    """单病例的全部指标 + 状态位。

    fast=True 时跳过 HD95 —— 只用于训练期选择 checkpoint，因为方案 §4.3 规定
    选择依据是「验证集三个场景的平均病例 Dice」，不需要 HD95。
    最终测试评价一律用 fast=False，HD95 照常计算。
    """
    p, g = pred > 0, gt > 0
    sp, sg = int(p.sum()), int(g.sum())
    m = {
        "dice": dice_foreground(p, g),
        "recall": recall_foreground(p, g),
        "hd95": float("nan") if fast else hd95(p, g, spacing, gt_cache),
        "pred_voxels": sp,
        "gt_voxels": sg,
        "pred_empty": sp == 0,
        "gt_empty": sg == 0,
        "complete_miss": bool(sg > 0 and sp == 0),   # 完全漏检
        "hd95_failed": bool(sg > 0 and sp == 0),     # HD95 无法有限计算
        "both_empty": bool(sg == 0 and sp == 0),
    }
    return m


def paired_bootstrap(diffs: np.ndarray, n: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """对逐病例配对差值做 bootstrap，给出均值差的 95% 区间（方案 §7.2）。

    只反映当前测试病例样本的不确定性；单次训练下不反映训练随机性。
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n, diffs.size))
    means = diffs[idx].mean(axis=1)
    return {
        "mean": float(diffs.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(diffs.size),
    }


def summarize(values) -> dict:
    v = np.asarray([x for x in values], dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v[ok].mean()), "std": float(v[ok].std(ddof=1)) if ok.sum() > 1 else 0.0,
            "n": int(ok.sum())}
