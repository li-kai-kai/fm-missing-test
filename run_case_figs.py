#!/usr/bin/env python3
"""病例可视化（方案 §7.3）。

  fig_cases.png      固定抽取的测试病例：可用 MRI / 真实 mask / A 预测 / B 预测
  fig_best_worst.png FM 最大改善与最大退步病例（明确标注为按结果选出）
  fig_fm_states.png  同一病例在 t=0,0.25,0.5,0.75,1 的连续状态（中间状态不是肿瘤生长）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import Config, MODALITIES, SCENARIOS, PRIMARY_SCENARIO
from fmexp.data import VolumeStore, avail_vector
from fmexp.infer import init_noise_for_case, predict_volume_A, predict_volume_B
from fmexp.metrics import dice_foreground
from fmexp.train import load_checkpoint
from fmexp.unet import build_model
from fmexp.viz import (AXIS, INK, MUTED, SERIES, SERIES_LABEL, SCEN_LABEL,
                       apply_style, best_slice, draw_slice, finish)


class Infer:
    def __init__(self, out, method, seed, ckpt, device):
        cfg = Config()
        ck = load_checkpoint(os.path.join(out, "runs", f"{method}_seed{seed}"), ckpt)
        self.model = build_model(method, cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.cfg = cfg
        self.method = method
        self.seed = seed


def predict(inf, case, scen, device, states=False):
    mri_full = torch.from_numpy(case.img).to(device)
    av = torch.from_numpy(avail_vector(scen)).to(device)
    mri = mri_full * av.view(4, 1, 1, 1)
    if inf.method == "A":
        pred = (predict_volume_A(inf.model, mri, av, inf.cfg) > 0.5).astype(np.uint8)
        return pred
    noise = init_noise_for_case(case.cid, case.seg.shape, inf.cfg.eval_seed, device)
    return predict_volume_B(inf.model, mri, av, inf.cfg, noise, collect_states=states)


def background_of(case, scen):
    """该场景下可用的第一个模态作为灰度底图（与推理输入一致）。"""
    av = SCENARIOS[scen]["avail"]
    for i, a in enumerate(av):
        if a:
            return case.img[i]
    return case.img[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--scenario", default=PRIMARY_SCENARIO)
    ap.add_argument("--n-cases", type=int, default=5)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    fig_dir = os.path.join(out, "figures")
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.load(open(os.path.join(out, "splits.json")))
    store = VolumeStore(os.path.join(out, "cache"), splits["test"])
    inf_a = Infer(out, "A", args.seed, args.ckpt, device)
    inf_b = Infer(out, "B", args.seed, args.ckpt, device)

    fixed = sorted(splits["test"])[:args.n_cases]
    res = {}
    for cid in fixed:
        case = store[cid]
        pa = predict(inf_a, case, args.scenario, device)
        pb = predict(inf_b, case, args.scenario, device)
        res[cid] = {"A": pa, "B": pb,
                    "dA": dice_foreground(pa, case.seg), "dB": dice_foreground(pb, case.seg)}
        print(f"{cid}: A={res[cid]['dA']:.4f} B={res[cid]['dB']:.4f}")

    # ---- 固定病例网格 ----
    ncol = 4
    fig, axes = plt.subplots(len(fixed), ncol, figsize=(2.5 * ncol, 2.55 * len(fixed)))
    axes = np.atleast_2d(axes)
    for r, cid in enumerate(fixed):
        case = store[cid]
        bg = background_of(case, args.scenario)
        idx = best_slice(case.seg, axis=2)
        draw_slice(axes[r, 0], bg, [], f"{cid.replace('BraTS20_Training_','')}  {SCEN_LABEL[args.scenario]}", idx=idx)
        draw_slice(axes[r, 1], bg, [(case.seg, "#ffffff", "GT")], "真实 mask（白 = 金标准轮廓）", idx=idx)
        draw_slice(axes[r, 2], bg, [(res[cid]["A"], SERIES["A"], "A")],
                   f"A 预测  Dice {res[cid]['dA']:.3f}", idx=idx)
        draw_slice(axes[r, 3], bg, [(res[cid]["B"], SERIES["B"], "B")],
                   f"B 预测  Dice {res[cid]['dB']:.3f}", idx=idx)
    fig.suptitle(f"固定抽取的测试病例（场景 {SCEN_LABEL[args.scenario]}，种子 {args.seed}）\n"
                 f"蓝 = A 直接分割，橙 = B FM 生成", y=1.0, color=INK, fontsize=10, fontweight="bold")
    fig.tight_layout()
    print("->", finish(fig, os.path.join(fig_dir, "fig_cases.png")))

    # ---- 最大改善 / 最大退步（按结果选出，明确标注）----
    mp = os.path.join(out, "metrics_per_case.csv")
    if os.path.exists(mp):
        rows = list(csv.DictReader(open(mp)))
        a = {r["patient_id"]: float(r["dice"]) for r in rows
             if r["model"] == "A" and r["scenario"] == args.scenario and int(r["seed"]) == args.seed}
        b = {r["patient_id"]: float(r["dice"]) for r in rows
             if r["model"] == "B" and r["scenario"] == args.scenario and int(r["seed"]) == args.seed}
        common = sorted(set(a) & set(b), key=lambda c: b[c] - a[c])
        if common:
            picks = [("最大退步", common[0]), ("最大改善", common[-1])]
            store2 = VolumeStore(os.path.join(out, "cache"), [c for _, c in picks])
            # 先把这两例的预测全部算好，再画图（固定 5 例之外的需要现算）
            for _, cid in picks:
                if cid not in res:
                    case = store2[cid]
                    pa = predict(inf_a, case, args.scenario, device)
                    pb = predict(inf_b, case, args.scenario, device)
                    res[cid] = {"A": pa, "B": pb}
            fig, axes = plt.subplots(len(picks), 4, figsize=(10, 2.55 * len(picks)))
            axes = np.atleast_2d(axes)
            for r, (tag, cid) in enumerate(picks):
                case = store2[cid]
                bg = background_of(case, args.scenario)
                idx = best_slice(case.seg, axis=2)
                draw_slice(axes[r, 0], bg, [(case.seg, "#ffffff", "GT")],
                           f"{tag}：{cid.replace('BraTS20_Training_','')}", idx=idx)
                draw_slice(axes[r, 1], bg, [], "可用 MRI（无轮廓）", idx=idx)
                draw_slice(axes[r, 2], bg,
                           [(case.seg, "#ffffff", "GT"), (res[cid]["A"], SERIES["A"], "A")],
                           f"A 预测  Dice {a[cid]:.3f}", idx=idx)
                draw_slice(axes[r, 3], bg,
                           [(case.seg, "#ffffff", "GT"), (res[cid]["B"], SERIES["B"], "B")],
                           f"B 预测  Dice {b[cid]:.3f}", idx=idx)
            fig.suptitle("按结果选出的病例（不是随机抽样）\n白=真实轮廓，蓝=A，橙=B",
                         y=1.0, color=INK, fontsize=10, fontweight="bold")
            fig.tight_layout()
            print("->", finish(fig, os.path.join(fig_dir, "fig_best_worst.png")))

    # ---- FM 中间状态 ----
    cid = fixed[0]
    case = store[cid]
    mri_full = torch.from_numpy(case.img).to(device)
    av = torch.from_numpy(avail_vector(args.scenario)).to(device)
    mri = mri_full * av.view(4, 1, 1, 1)
    noise = init_noise_for_case(cid, case.seg.shape, inf_b.cfg.eval_seed, device)
    out_b = predict_volume_B(inf_b.model, mri, av, inf_b.cfg, noise, collect_states=True)
    states, final_pred = out_b["states"], out_b["pred"]
    ts = sorted(states)
    bg = background_of(case, args.scenario)
    idx = best_slice(case.seg, axis=2)
    # 所有时步共用同一色标，否则看不出演化
    vmax = max(float(np.percentile(np.abs(s[tuple([slice(None)] * 2 + [idx])]), 99.5)) for s in states.values())
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, len(ts) + 1, figsize=(2.3 * (len(ts) + 1), 2.9))
    axes = np.atleast_1d(axes)
    bgslice = np.rot90(bg[:, :, idx])
    for ax, t in zip(axes, ts):
        s = np.rot90(states[t][:, :, idx])
        ax.imshow(bgslice, cmap="gray", interpolation="nearest")
        ax.imshow(np.ma.masked_where(np.abs(s) < 0.05 * vmax, s), cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, alpha=0.85, interpolation="nearest")
        gt = np.rot90(case.seg[:, :, idx].astype(float))
        if gt.max() > 0:
            ax.contour(gt, levels=[0.5], colors=["#000000"], linewidths=1.0)
        ax.set_title(f"t = {t:g}", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color(AXIS); sp.set_linewidth(0.6)
    # 最后一步：真正离散化后的预测
    ax = axes[-1]
    ax.imshow(bgslice, cmap="gray", interpolation="nearest")
    ax.contour(np.rot90(final_pred[:, :, idx].astype(float)), levels=[0.5],
               colors=[SERIES["B"]], linewidths=1.4)
    ax.set_title("t = 1 的 argmax 预测", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(AXIS); sp.set_linewidth(0.6)
    fig.suptitle(
        f"{cid.replace('BraTS20_Training_','')} 的采样过程（场景 {args.scenario}，{inf_b.cfg.fm_steps} 步 Euler）\n"
        f"彩色 = 连续状态 y[1]−y[0]（红 = 偏前景）；黑线 = 真实轮廓；只有 t=1 才有离散预测。\n"
        f"中间状态不是肿瘤随时间的生长，中途 argmax 没有意义。",
        y=1.06, color=INK, fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    print("->", finish(fig, os.path.join(fig_dir, "fig_fm_states.png")))


if __name__ == "__main__":
    main()
