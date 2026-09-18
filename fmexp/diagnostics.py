"""失败归因诊断的度量函数（只读，不训练，不改动冻结配置）。

四类检查对应四组函数：
  1. 拟合      —— 训练病例 / 小样本上的完整生成质量
  2. 时间与条件 —— 分时间的速度 MSE、理想路径 vs 实际轨迹、MRI 条件消融、终点外推
  3. 错误位置   —— 脑内/脑外假阳、类别混淆、脑外假阳到脑支撑的距离、连通块
  4. 步数       —— Euler 步数扫描

约定：
  * 本模块所有测量都允许读取真实标签，但真实标签**绝不进入模型输入**。
  * 诊断性推理变体（掩蔽、打乱 MRI、置零 MRI）只是读数，不是方法结果。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

from .config import Config
from .data import avail_vector
from .infer import _velocity_field, build_input_A, plan_windows, predict_volume_B


# ============================================================ 通用小工具

def onehot_from_seg(seg: np.ndarray, n_classes: int) -> np.ndarray:
    """[D,H,W] 标签 -> [K,D,H,W] one-hot float32。"""
    return np.eye(n_classes, dtype=np.float32)[seg].transpose(3, 0, 1, 2)


def _mean_sq(a: np.ndarray) -> float:
    return float(np.mean(np.square(a)))


def connected_component_stats(mask: np.ndarray) -> Dict[str, float]:
    """连通块统计。6 邻域。`small_frac` = 小于最大块 1% 的碎块所占体素比例。"""
    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total == 0:
        return {"n_components": 0, "largest": 0, "small_frac": 0.0, "total_voxels": 0}
    lab, n = ndimage.label(m)
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())
    small = float(sizes[sizes < 0.01 * largest].sum()) / total
    return {"n_components": int(n), "largest": largest,
            "small_frac": small, "total_voxels": total}


# ============================================================ 检查 3：错误位置

def error_location(pred: np.ndarray, gt: np.ndarray, support: np.ndarray,
                   n_classes: int = 4, dist_to_support: np.ndarray | None = None) -> Dict:
    """把错误拆成脑内/脑外，并给出类别混淆与碎块统计。

    dist_to_support: 预计算的「到最近脑支撑体素的距离」（每例算一次即可复用）。
    """
    pf, gf = pred > 0, gt > 0
    tp = int((pf & gf).sum())
    fn = int((~pf & gf).sum())
    fp_in = int((pf & ~gf & support).sum())
    fp_out = int((pf & ~gf & ~support).sum())
    fp = fp_in + fp_out
    out = {
        "tp": tp, "fn": fn, "fp_in": fp_in, "fp_out": fp_out,
        "gt_voxels": int(gf.sum()), "support_voxels": int(support.sum()),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "fp_out_frac": fp_out / max(fp, 1),
        "pred_out_frac": int((pf & ~support).sum()) / max(int(pf.sum()), 1),
        "components": connected_component_stats(pf),
    }
    if n_classes > 2:
        m = gf
        cm = np.bincount((gt[m].astype(np.int64) * n_classes + pred[m].astype(np.int64)).ravel(),
                         minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        out["confusion"] = cm.tolist()
    if dist_to_support is not None and fp_out > 0:
        d = dist_to_support[pf & ~gf & ~support]
        out["fp_out_distance"] = {
            "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "max": float(d.max()),
            "within_5mm": float((d <= 5).mean()), "within_10mm": float((d <= 10).mean()),
        }
    return out


def distance_to_support(support: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """每个体素到最近脑支撑体素的距离（mm）。每例算一次，可跨场景复用。"""
    return ndimage.distance_transform_edt(~np.asarray(support, dtype=bool), sampling=spacing)


def mask_to_support(pred: np.ndarray, support: np.ndarray) -> np.ndarray:
    """诊断读数：脑外一律判为背景。不是方法结果，只用于量化 OOD 窗口的损失。"""
    out = pred.copy()
    out[~support] = 0
    return out


# ============================================================ 检查 2a/2b：速度 MSE

def _velocity_at(model, y: torch.Tensor, t: float, cond_win, slices, wgt, shape,
                 cfg: Config, chunk: int = 32) -> torch.Tensor:
    return _velocity_field(model, y, t, cond_win, slices, wgt, shape, cfg, chunk=chunk)


@torch.no_grad()
def velocity_mse_by_t(model, store, cases: Sequence[str], cfg: Config, device,
                      t_edges: Sequence[float] = (0.0, 0.125, 0.25, 0.375, 0.5,
                                                  0.625, 0.75, 0.875, 1.0),
                      n_batches: int = 8, seed: int = 0,
                      chunk: int = 32) -> Dict:
    """在 patch 上、沿**理想路径** `(1-t)y0 + t*y1` 统计速度 MSE。

    返回每个 t 分箱的：
      mse        模型 / 常数预测器 / 独立先验 Bayes 参考
      r2         1 - mse_model / var_v
      per_class  按真实类别分组的模型 MSE 与体素数
    """
    model.eval()
    edges = np.asarray(t_edges, dtype=float)
    K = cfg.n_classes
    acc = {i: {"se": 0.0, "n": 0, "v_se": 0.0} for i in range(len(edges) - 1)}
    cls = {i: {c: {"se": 0.0, "n": 0} for c in range(K)} for i in range(len(edges) - 1)}
    # patch 尺寸等于滑窗尺寸时 plan_windows 只返回一个窗口，等价于一次普通前向；
    # _velocity_field 本来就是对单个（无 batch 维）体积写的，这里直接沿用同一形状。
    patches = []
    for b in range(n_batches):
        batch = make_patches(store, cases, seed, b, cfg, device)
        for i in range(batch["y1"].shape[0]):
            patches.append((batch["mri"][i], batch["avail"][i], batch["y1"][i]))
    gen = torch.Generator(device="cpu").manual_seed(12345 + seed)
    for mri, av, pri in patches:
        shape = tuple(mri.shape[-3:])
        slices, wgt = plan_windows(shape, cfg.sw_window, cfg.sw_overlap)
        wgt = wgt.to(device)
        cond = build_input_A(mri, av)
        cond_win = torch.empty((len(slices), 8, cfg.sw_window, cfg.sw_window, cfg.sw_window),
                               dtype=torch.float32, device=device)
        for i, sl in enumerate(slices):
            cond_win[i] = cond[(slice(None),) + sl]
        lab = pri.argmax(dim=0)
        for k in range(len(edges) - 1):
            t = float(0.5 * (edges[k] + edges[k + 1]))
            y0 = torch.randn((K,) + shape, generator=gen, dtype=torch.float32).to(device)
            yt = (1 - t) * y0 + t * pri
            v_true = pri - y0
            v_pred = _velocity_at(model, yt, t, cond_win, slices, wgt, shape, cfg, chunk=chunk)
            err = v_pred - v_true
            acc[k]["se"] += float((err ** 2).sum())
            acc[k]["n"] += err.numel()
            acc[k]["v_se"] += float(((v_true - v_true.mean()) ** 2).sum())
            for c in range(K):
                m = lab == c                       # [p,p,p]，err 是 [K,p,p,p]
                if bool(m.any()):
                    cls[k][c]["se"] += float((err[:, m] ** 2).sum())
                    cls[k][c]["n"] += int(m.sum()) * K
    out = {"t_edges": edges.tolist(), "bins": []}
    prior = _prior_from_store(store, cases, cfg, device)
    for k in range(len(edges) - 1):
        n = max(acc[k]["n"], 1)
        mse = acc[k]["se"] / n
        var = acc[k]["v_se"] / n
        bayes = bayes_velocity_mse(0.5 * (edges[k] + edges[k + 1]), prior)
        out["bins"].append({
            "t": float(0.5 * (edges[k] + edges[k + 1])),
            "mse_model": mse, "var_v": var, "mse_prior_only_ref": bayes,
            "r2": 1.0 - mse / var if var > 1e-12 else float("nan"),
            "ratio_to_prior_only": mse / bayes if bayes > 1e-12 else float("nan"),
            "per_class": {c: {"mse": cls[k][c]["se"] / max(cls[k][c]["n"], 1),
                              "n": cls[k][c]["n"]} for c in range(K)},
        })
    return out


def make_patches(store, cases, seed, step, cfg: Config, device):
    """抽一个不带增强的干净 batch；返回 torch 张量。"""
    from .data import make_batch
    b = make_batch(store, cases, seed, step, cfg.patch, cfg.tumor_center_prob,
                   cfg.batch_size, augment=False, n_classes=cfg.n_classes,
                   scenario_order=cfg.scenarios)
    return {
        "mri": torch.from_numpy(b["mri"]).to(device),
        "y1": torch.from_numpy(b["y1"]).to(device),
        "seg": b["y1"].argmax(axis=1).astype(np.uint8),
        "avail": torch.from_numpy(b["avail"]).to(device),
    }


_PRIOR_CACHE: Dict[Tuple, np.ndarray] = {}


def _prior_from_store(store, cases, cfg: Config, device) -> np.ndarray:
    """用训练 patch 分布估计类别先验（只依赖配置与病例集，缓存复用）。"""
    key = (tuple(cases), cfg.patch, cfg.batch_size, cfg.task)
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]
    cnt = np.zeros(cfg.n_classes, dtype=np.float64)
    for step in range(4):
        b = make_patches(store, cases, 0, step, cfg, device)
        for lab in b["seg"].ravel():
            cnt[lab] += 1
    p = cnt / cnt.sum()
    _PRIOR_CACHE[key] = p
    return p


def bayes_velocity_mse(t: float, prior: np.ndarray, n: int = 200_000,
                       gen: np.random.Generator | None = None) -> float:
    """「只知道全局类别先验」时的最优速度 MSE —— 参考线，**不是下界**。

    对固定标签 y1，给定 y_t 后 y0=(y_t−t·y1)/(1−t) 被完全确定，因此最优速度
    v*=(y1−y_t)/(1−t) 的残差为 0；真正的困难只在于从 y_t 反推 y1。
    本函数把 y1 的先验取成全局类别频率且**假设体素独立**，于是后验为
      log P(c|y) ∝ log π_c − ‖y − t·e_c‖²/(2(1−t)²)，v* = (E[y1|y] − y)/(1−t)。

    真实模型能用空间上下文和 MRI，**必须显著低于**这条线；低 t 处若模型的
    误差贴近这条线，说明它在该阶段几乎没用到条件信息。
    """
    if t >= 1.0:
        t = 1.0 - 1e-6
    rng = gen or np.random.default_rng(0)
    prior = np.asarray(prior, dtype=np.float64)
    prior = np.maximum(prior, 1e-12)
    prior = prior / prior.sum()
    K = len(prior)
    c = rng.choice(K, size=n, p=prior)
    y1 = np.eye(K, dtype=np.float32)[c]
    y0 = rng.standard_normal((n, K)).astype(np.float32)
    yt = (1 - t) * y0 + t * y1
    # yt[:,None,:] 对每个候选类别 e_c 求 ‖y − t·e_c‖²，得到 [n,K] 的后验 logits
    logits = (np.log(prior)[None, :]
              - ((yt[:, None, :] - t * np.eye(K)[None, :]) ** 2).sum(-1) / (2 * (1 - t) ** 2))
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    q /= q.sum(axis=1, keepdims=True)
    v_star = (q - yt) / (1 - t)
    return _mean_sq(v_star - (y1 - y0))


# ============================================================ 检查 2b：轨迹

@torch.no_grad()
def trajectory_diagnostic(model, mri: torch.Tensor, avail: torch.Tensor, cfg: Config,
                          init_noise: torch.Tensor, seg: np.ndarray,
                          steps: int | None = None, chunk: int = 32) -> Dict:
    """量化「误差是否复利」：同一批 t 上比较理想路径点与实际轨迹点。

    真实速度场是常数 v_true = y1 − y0，因此两个位置上的速度 MSE 可直接比较：
      * v_mse_onideal —— 把模型放在理想路径点上问（场本身准不准）
      * v_mse_ontraj —— 放在模型自己积分出来的点上问（采样实际访问到的位置）
    另外记录轨迹偏离理想路径的距离 drift，以及各步 argmax 的 Dice。
    """
    model.eval()
    dev = mri.device
    steps = steps or cfg.fm_steps
    K = cfg.n_classes
    shape = tuple(mri.shape[-3:])
    slices, wgt = plan_windows(shape, cfg.sw_window, cfg.sw_overlap)
    wgt = wgt.to(dev)
    cond = build_input_A(mri, avail)
    cond_win = torch.empty((len(slices), 8, cfg.sw_window, cfg.sw_window, cfg.sw_window),
                           dtype=torch.float32, device=dev)
    for i, sl in enumerate(slices):
        cond_win[i] = cond[(slice(None),) + sl]

    y0 = init_noise.to(dev).float().clone()
    y1 = torch.from_numpy(onehot_from_seg(seg, K)).to(dev)
    v_true = y1 - y0
    var_v = float(((v_true - v_true.mean()) ** 2).mean())
    dt = 1.0 / steps
    y = y0.clone()
    rows = []
    for k in range(steps):
        t = k / steps
        ideal = (1 - t) * y0 + t * y1
        v_pred = _velocity_at(model, y, t, cond_win, slices, wgt, shape, cfg, chunk=chunk)
        v_ideal = _velocity_at(model, ideal, t, cond_win, slices, wgt, shape, cfg, chunk=chunk)
        pred_k = y.argmax(dim=0).cpu().numpy().astype(np.uint8)
        rows.append({
            "step": k, "t": t,
            "drift_mse": float(((y - ideal) ** 2).mean()),
            "v_mse_ontraj": float(((v_pred - v_true) ** 2).mean()),
            "v_mse_onideal": float(((v_ideal - v_true) ** 2).mean()),
            "dice_traj": _region_dice(pred_k, seg, "WT"),
            "dice_ideal": _region_dice(ideal.argmax(dim=0).cpu().numpy().astype(np.uint8), seg, "WT"),
        })
        y = y + dt * v_pred
    final = y.argmax(dim=0).cpu().numpy().astype(np.uint8)
    return {"steps": steps, "var_v": var_v, "rows": rows,
            "final_dice": {r: _region_dice(final, seg, r) for r in ("WT", "TC", "ET")}}


def _region_masks(seg: np.ndarray) -> Dict[str, np.ndarray]:
    return {"WT": seg > 0, "TC": (seg == 1) | (seg == 3), "ET": seg == 3}


def _region_dice(a: np.ndarray, b: np.ndarray, region: str) -> float:
    pa, pb = _region_masks(a)[region], _region_masks(b)[region]
    sa, sb = int(pa.sum()), int(pb.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    return float(2.0 * (pa & pb).sum() / (sa + sb))


# ============================================================ 检查 2c：条件消融

def conditioning_variants(mri_full: np.ndarray, support: np.ndarray, cid: str,
                          other_mri: np.ndarray, eval_seed: int,
                          rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """构造四种 MRI 条件，返回 {变体名: MRI[4,D,H,W]}。

    real       真实 MRI
    zero       四通道全置零（存在标记不变）
    swap_case  换成另一个病例的 MRI（解剖错误、统计特性正确）
    shuffle_vox 脑内体素随机重排（保留强度边缘分布、破坏空间结构）
    """
    real = mri_full
    zero = np.zeros_like(mri_full)
    swap = other_mri
    shuf = np.zeros_like(mri_full)
    idx = np.argwhere(support)
    perm = rng.permutation(len(idx))
    shuf[(slice(None),) + tuple(idx.T)] = real[(slice(None),) + tuple(idx[perm].T)]
    return {"real": real, "zero": zero, "swap_case": swap, "shuffle_vox": shuf}


# ============================================================ 检查 4：步数扫描

@torch.no_grad()
def steps_sweep(model, mri: torch.Tensor, avail: torch.Tensor, cfg: Config,
                init_noise: torch.Tensor, seg: np.ndarray,
                steps_list: Sequence[int], chunk: int = 32) -> Dict[int, Dict]:
    """固定 checkpoint 与噪声，比较不同 Euler 步数。一并记录连通块数。"""
    out = {}
    for s in steps_list:
        pred = predict_volume_B(model, mri, avail, cfg, init_noise, steps=s, chunk=chunk)
        out[int(s)] = {
            "dice": {r: _region_dice(pred, seg, r) for r in ("WT", "TC", "ET")},
            "components": connected_component_stats(pred > 0),
            "pred_voxels": int((pred > 0).sum()),
        }
    return out


# ============================================================ 检查 2e：端点置信间隔

@torch.no_grad()
def final_state(model, mri: torch.Tensor, avail: torch.Tensor, cfg: Config,
                init_noise: torch.Tensor, steps: int | None = None,
                chunk: int = 32) -> torch.Tensor:
    """跑完整积分并返回**最终的连续状态 y**（不解 argmax）。

    与 predict_volume_B 逐步一致，只是把末态留下来。one-hot 目标的值域是
    「正确类≈1、其余≈0」，因此 max−second 就是该体素的判定置信间隔。
    """
    model.eval()
    dev = mri.device
    steps = steps or cfg.fm_steps
    shape = tuple(mri.shape[-3:])
    slices, wgt = plan_windows(shape, cfg.sw_window, cfg.sw_overlap)
    wgt = wgt.to(dev)
    cond = build_input_A(mri, avail)
    cond_win = torch.empty((len(slices), 8, cfg.sw_window, cfg.sw_window, cfg.sw_window),
                           dtype=torch.float32, device=dev)
    for i, sl in enumerate(slices):
        cond_win[i] = cond[(slice(None),) + sl]
    y = init_noise.to(dev).float().clone()
    dt = 1.0 / steps
    for k in range(steps):
        v = _velocity_field(model, y, k / steps, cond_win, slices, wgt, shape, cfg, chunk=chunk)
        y = y + dt * v
    return y


def endpoint_margin(y: torch.Tensor, seg: np.ndarray, support: np.ndarray) -> Dict:
    """终点连续状态的置信间隔统计。y: [K,D,H,W]。"""
    top2 = y.topk(2, dim=0).values
    margin = (top2[0] - top2[1]).cpu().numpy()
    pred = y.argmax(dim=0).cpu().numpy().astype(np.uint8)
    correct = pred == seg
    tot = y.sum(dim=0).cpu().numpy()
    out = {
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.percentile(margin, 10)),
        "frac_margin_lt_0.1": float((margin < 0.1).mean()),
        "frac_margin_lt_0.25": float((margin < 0.25).mean()),
        "frac_margin_lt_0.5": float((margin < 0.5).mean()),
        "margin_correct_mean": float(margin[correct].mean()) if correct.any() else None,
        "margin_wrong_mean": float(margin[~correct].mean()) if (~correct).any() else None,
        "channelsum_mean": float(tot.mean()),
        "channelsum_std": float(tot.std()),
        "y_min": float(y.min()), "y_max": float(y.max()),
    }
    # 脑内 / 脑外的间隔分布，用于判断「糊掉」发生在哪里
    for name, m in (("in_brain", support), ("out_brain", ~support)):
        if m.any():
            out[f"margin_median_{name}"] = float(np.median(margin[m]))
            out[f"frac_lt_0.25_{name}"] = float((margin[m] < 0.25).mean())
    return out
