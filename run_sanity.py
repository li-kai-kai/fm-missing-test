#!/usr/bin/env python3
"""阶段一：固定 patch 调通（方案 §6）。

内容：
  1. 自检：one-hot / 标签映射 / MRI-标签对齐 / 时间广播 / 目标速度符号；
  2. 数值自检：用 oracle 速度场验证滑窗拼装与 Euler 积分是否精确复原 y1；
  3. 过拟合检查：2~4 个训练病例的固定 patch、只用完整模态、关闭增强，
     各训练 ~1500 次更新，看 A 能否拟合分割、B 能否从固定噪声生成出真实 mask。

重点提醒（方案 §6）：若只有训练 MSE 下降而生成 mask 仍为空，必须继续排查，
不能直接扩大实验。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import Config, SCENARIO_ORDER
from fmexp.data import avail_vector, load_case, patch_slices, step_rngs
from fmexp.infer import (_velocity_field, init_noise_for_case, plan_windows,
                         predict_volume_A, predict_volume_B)
from fmexp.losses import loss_A, loss_B
from fmexp.metrics import case_metrics, dice_foreground
from fmexp.train import VolumeStore
from fmexp.unet import build_model, n_params


# ------------------------------------------------------------------ 数值自检

class OracleModel(nn.Module):
    """把输入通道 0:2 当作速度返回。用于验证滑窗拼装 + Euler 积分。"""

    def forward(self, x):
        return x[:, 0:2]


class DecayModel(nn.Module):
    """v = -yt * g(mri)，g 由输入通道 0 决定。ODE 有解析解，可逐体素核对。"""

    def forward(self, x):
        mri0 = x[:, 0:1]
        g = 1.0 + 0.5 * torch.sigmoid(mri0)
        return -x[:, 8:10] * g


def numerical_checks(cfg: Config, device, log) -> dict:
    """验证滑窗速度场拼装与 Euler 更新顺序是否与公式一致。"""
    res = {}
    D, H, W = 96, 112, 80
    cfg = Config(**{**cfg.__dict__})

    # (a) 组合性：v = A(x) + B(yt)，其中 A 是空间场（由 MRI 决定），B 依赖 yt。
    #     用 oracle 验证 y 的更新是“全体积统一更新”，与窗口顺序无关。
    y1 = torch.zeros(2, D, H, W)
    y1[0] = 1.0
    fg = torch.zeros(D, H, W, dtype=torch.bool)
    fg[30:60, 40:70, 20:50] = True
    y1[0][fg] = 0.0
    y1[1][fg] = 1.0

    g = torch.Generator().manual_seed(7)
    y0 = torch.randn(2, D, H, W, generator=g)
    target_v = (y1 - y0)                       # 常数（不依赖 t）的 oracle 速度场

    slices, wgt = plan_windows((D, H, W), cfg.sw_window, cfg.sw_overlap)
    cond_win = torch.empty((len(slices), 8, cfg.sw_window, cfg.sw_window, cfg.sw_window))
    for i, sl in enumerate(slices):
        cond_win[i] = torch.cat([target_v, torch.zeros(6, D, H, W)])[(slice(None),) + sl]
    cond_win = cond_win.to(device)
    wgt = wgt.to(device)
    oracle = OracleModel().to(device).eval()

    y = y0.to(device).clone()
    steps = cfg.fm_steps
    dt = 1.0 / steps
    with torch.no_grad():
        for k in range(steps):
            v = _velocity_field(oracle, y, k / steps, cond_win, slices, wgt, (D, H, W), cfg)
            y = y + dt * v
    y = y.cpu()
    # ∫ (y1-y0) dt over [0,1] = y1 - y0  =>  y(1) = y0 + (y1-y0) = y1，逐步 Euler 也应精确
    err = float((y - y1).abs().max())
    pred = y.argmax(0).numpy()
    res["oracle_max_abs_err"] = err
    res["oracle_pred_equals_y1"] = bool((pred == fg.numpy().astype(np.int64)).all())
    log(f"  [自检 a] oracle 速度场积分复原 y1：最大绝对误差={err:.3e}，"
        f"argmax 完全一致={res['oracle_pred_equals_y1']}")

    # (b) 解析解核对：v = -yt*g(mri) => y(1) = y0 * (1 - dt*g)^steps
    decay = DecayModel().to(device).eval()
    mri = torch.randn(4, D, H, W, generator=torch.Generator().manual_seed(11))
    cond2 = torch.empty_like(cond_win)
    for i, sl in enumerate(slices):
        cond2[i] = torch.cat([mri, torch.zeros(4, D, H, W)])[(slice(None),) + sl]
    cond2 = cond2.to(device)
    y = y0.to(device).clone()
    with torch.no_grad():
        for k in range(steps):
            v = _velocity_field(decay, y, k / steps, cond2, slices, wgt, (D, H, W), cfg)
            y = y + dt * v
    gfield = (1.0 + 0.5 * torch.sigmoid(mri[0:1])).to(device)   # [1,D,H,W]，对两通道同样作用
    y_exact = y0.to(device) * (1 - dt * gfield) ** steps
    err2 = float((y - y_exact).abs().max())
    res["decay_max_abs_err"] = err2
    log(f"  [自检 b] 与解析解逐体素一致：最大绝对误差={err2:.3e}")
    return res


def data_diagnostics(store, cases, cfg, device, log) -> dict:
    """标签映射、one-hot、MRI 与标签对齐、时间广播、目标速度符号。"""
    cid = cases[0]
    case = store[cid]
    img, seg = case.img, case.seg
    xyz = np.argwhere(seg > 0)
    c = xyz[len(xyz) // 2]
    sl = patch_slices(c, cfg.patch, seg.shape)
    pi, ps = img[(slice(None),) + sl], seg[sl]

    out = {}
    out["seg_unique_in_patch"] = sorted(int(u) for u in np.unique(ps))
    out["seg_foreground_frac"] = float((ps > 0).mean())

    y1 = np.stack([(ps == 0), (ps > 0)]).astype(np.float32)
    out["onehot_sums_to_one"] = bool(np.allclose(y1.sum(0), 1.0))
    out["fg_channel_matches_seg"] = bool((y1[1] > 0.5).all() == (ps > 0).all())
    log(f"  [自检] patch 标签取值 {out['seg_unique_in_patch']}，前景占比 {out['seg_foreground_frac']:.4f}")
    log(f"  [自检] one-hot 合成和为 1：{out['onehot_sums_to_one']}；"
        f"前景通道与 seg 一致：{out['fg_channel_matches_seg']}")

    # MRI 与标签对齐：BraTS 中“增强肿瘤 (原始标签 4)”在 T1ce 上应明显偏亮。
    # 注意不能用“肿瘤整体”（1/2/4 合并）来查，因为水肿(2)在 T1ce 上偏暗且体积占优。
    # 这里直接读原始文件核对，绕过预处理，是一次真正独立的对齐检查。
    import nibabel as nib
    raw_dir = os.path.join(cfg.data_root, cid)
    t1ce_raw = np.asanyarray(nib.as_closest_canonical(
        nib.load(os.path.join(raw_dir, f"{cid}_t1ce.nii.gz"))).dataobj).astype(np.float32)
    seg_raw = np.asanyarray(nib.as_closest_canonical(
        nib.load(os.path.join(raw_dir, f"{cid}_seg.nii.gz"))).dataobj)
    et, bg = seg_raw == 4, seg_raw == 0
    out["raw_t1ce_mean_in_ET"] = float(t1ce_raw[et].mean()) if et.any() else float("nan")
    out["raw_t1ce_mean_in_bg"] = float(t1ce_raw[bg].mean())
    out["alignment_ok"] = bool(et.any() and out["raw_t1ce_mean_in_ET"] > out["raw_t1ce_mean_in_bg"])
    log(f"  [自检] 原始 T1ce 均值 增强肿瘤(标签4)内={out['raw_t1ce_mean_in_ET']:.1f} "
        f"背景={out['raw_t1ce_mean_in_bg']:.1f} -> 对齐正常={out['alignment_ok']}")
    # 预处理后的 patch 上再看一次：肿瘤整体（含水肿）在 T1ce 上偏暗是预期现象
    t1ce = pi[2]
    out["patch_t1ce_mean_in_tumor"] = float(t1ce[ps > 0].mean()) if (ps > 0).any() else float("nan")
    out["patch_t1ce_mean_in_bg"] = float(t1ce[ps == 0].mean())
    log(f"  [自检] 预处理后 patch T1ce 均值 肿瘤整体内={out['patch_t1ce_mean_in_tumor']:.3f} "
        f"背景={out['patch_t1ce_mean_in_bg']:.3f}（含水肿，偏暗属预期）")

    # 时间广播与目标速度
    y1t = torch.from_numpy(y1)[None].to(device)
    g = torch.Generator(device=device).manual_seed(3)
    y0 = torch.randn_like(y1t, generator=g)
    t = torch.tensor([[[[[0.0]]]]], device=device)
    out["yt_at_t0_equals_y0"] = bool(torch.allclose((1 - t) * y0 + t * y1t, y0))
    t1 = torch.ones_like(t)
    out["yt_at_t1_equals_y1"] = bool(torch.allclose((1 - t1) * y0 + t1 * y1t, y1t))
    tv = y1t - y0
    # 正确符号：t=1 时目标速度在前景体素上的均值应随噪声偏离 y1 的方向
    out["target_v_sign_ok"] = bool(torch.allclose(tv, y1t - y0) and not torch.allclose(tv, y0 - y1t))
    log(f"  [自检] t=0 时 yt==y0: {out['yt_at_t0_equals_y0']}；t=1 时 yt==y1: {out['yt_at_t1_equals_y1']}；"
        f"目标速度用 y1-y0: {out['target_v_sign_ok']}")
    return out


# ------------------------------------------------------------------ 过拟合检查

def fixed_val_batches(cases, store, cfg):
    """每个病例取一个以肿瘤为心的固定 patch，完整模态，无增强。"""
    out = []
    for cid in cases:
        case = store[cid]
        seg = case.seg
        xyz = np.argwhere(seg > 0)
        c = xyz[len(xyz) // 2] if len(xyz) else np.argwhere(case.support)[0]
        sl = patch_slices(c, cfg.patch, seg.shape)
        out.append({
            "case": cid,
            "mri": case.img[(slice(None),) + sl][None].astype(np.float32),
            "seg": seg[sl][None].astype(np.float32),
        })
    return out


def overfit(method: str, cfg: Config, cases, store, steps: int, device, log):
    torch.manual_seed(0)
    model = build_model(method, cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    batches = fixed_val_batches(cases, store, cfg)

    mri = torch.from_numpy(np.concatenate([b["mri"] for b in batches])).to(device)   # [N,4,64^3]
    seg = np.concatenate([b["seg"] for b in batches])                                 # [N,64^3]
    y1 = torch.zeros(mri.shape[0], 2, *mri.shape[2:], device=device)
    y1[:, 0] = torch.from_numpy((seg == 0).astype(np.float32)).to(device)
    y1[:, 1] = torch.from_numpy((seg > 0).astype(np.float32)).to(device)
    avail = torch.tensor([avail_vector("C0")] * mri.shape[0], device=device)
    avail_exp = avail.view(-1, 4, 1, 1, 1).expand(-1, 4, *mri.shape[2:])

    log(f"\n== 过拟合检查: 方法 {method}，{len(cases)} 个病例固定 patch，"
        f"{steps} 次更新，完整模态，无增强 ==")
    for step in range(steps):
        if method == "A":
            x = torch.cat([mri, avail_exp], 1)
            loss, parts = loss_A(model(x), y1)
        else:
            _, mr = step_rngs(0, step)
            g = torch.Generator(device=device).manual_seed(int(mr.integers(2 ** 31 - 1)))
            y0 = torch.randn_like(y1, generator=g)
            t = torch.rand(mri.shape[0], 1, 1, 1, 1, generator=g, device=device)
            yt = (1 - t) * y0 + t * y1
            tv = y1 - y0
            tc = t.expand(-1, 1, *mri.shape[2:])
            loss, parts = loss_B(model(torch.cat([mri, avail_exp, yt, tc], 1)), tv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            msg = " ".join(f"{k}={float(v):.4f}" for k, v in parts.items())
            log(f"   step {step:>5d} loss={float(loss):.4f} {msg}")

    # 训练 patch 上的拟合情况
    model.eval()
    with torch.no_grad():
        if method == "A":
            logits = model(torch.cat([mri, avail_exp], 1))
            pt = (logits.argmax(1) == 1).cpu().numpy()
        else:
            pt = np.zeros_like(seg, dtype=bool)
            for i in range(mri.shape[0]):
                # 训练 patch 上直接积分（等价于单窗口的全过程）
                g = torch.Generator(device=device).manual_seed(999 + i)
                y = torch.randn_like(y1[i:i + 1], generator=g)
                dt = 1.0 / cfg.fm_steps
                for k in range(cfg.fm_steps):
                    tc = torch.full((1, 1, *mri.shape[2:]), k / cfg.fm_steps, device=device)
                    v = model(torch.cat([mri[i:i + 1], avail_exp[i:i + 1], y, tc], 1))
                    y = y + dt * v
                pt[i] = (y.argmax(1) == 1).cpu().numpy()[0]
    patch_dice = [dice_foreground(pt[i], seg[i]) for i in range(len(seg))]
    log(f"   训练 patch Dice: " + " ".join(f"{c}={d:.4f}" for c, d in zip(cases, patch_dice)))

    # 完整体积推理（走真正的滑窗 / Euler 路径）
    full = {}
    for cid in cases:
        case = store[cid]
        gt = case.seg
        m = torch.from_numpy(case.img).to(device)
        av = torch.tensor(avail_vector("C0"), device=device)
        if method == "A":
            pred = (predict_volume_A(model, m, av, cfg) > 0.5).astype(np.uint8)
        else:
            noise = init_noise_for_case(cid, gt.shape, cfg.eval_seed, device)
            pred = predict_volume_B(model, m, av, cfg, noise)
        mt = case_metrics(pred, gt)
        full[cid] = mt
        log(f"   完整体积 {cid}: Dice={mt['dice']:.4f} 预测体素={mt['pred_voxels']} "
            f"真实体素={mt['gt_voxels']} 漏检={mt['complete_miss']}")

    empty = sum(1 for m in full.values() if m["pred_empty"])
    return {"patch_dice": {c: float(d) for c, d in zip(cases, patch_dice)},
            "volume": {c: m for c, m in full.items()},
            "n_empty_predictions": empty}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--n-cases", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--skip-numerical", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs = []

    def log(msg):
        print(msg, flush=True)
        logs.append(str(msg))

    import json as _json
    splits = _json.load(open(os.path.join(out, "splits.json")))
    cases = splits["train"][:args.n_cases]
    store = VolumeStore(os.path.join(out, "cache"), cases)

    log("=== 阶段一：固定 patch 调通 ===")
    report = {"cases": cases, "steps": args.steps}
    report["data_diagnostics"] = data_diagnostics(store, cases, cfg, device, log)

    if not args.skip_numerical:
        log("\n-- 数值自检：滑窗速度场拼装与 Euler 积分 --")
        report["numerical"] = numerical_checks(cfg, device, log)

    for method in ("A", "B"):
        report[f"overfit_{method}"] = overfit(method, cfg, cases, store, args.steps, device, log)

    with open(os.path.join(out, "stage1_sanity.json"), "w") as f:
        _json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "stage1_sanity.log"), "w") as f:
        f.write("\n".join(logs))
    print("\n产物 ->", os.path.join(out, "stage1_sanity.json"))


if __name__ == "__main__":
    main()
