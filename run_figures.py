#!/usr/bin/env python3
"""训练曲线与指标图（方案 §7.2、§7.3）。"""
from __future__ import annotations

import argparse
import csv
import json
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import SCENARIO_ORDER, PRIMARY_SCENARIO
from fmexp.metrics import paired_bootstrap
from fmexp.viz import (ACCENT, AXIS, GRID, INK, INK2, MUTED, SERIES, SERIES_LABEL,
                       SCEN_LABEL, apply_style, finish, _strip)


def read_val_curves(runs_dir, methods, seeds):
    curves = {}
    for m in methods:
        for s in seeds:
            p = os.path.join(runs_dir, f"{m}_seed{s}", "log.jsonl")
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p) if l.strip()]
            rows = [r for r in rows if r["type"] == "val"]
            if rows:
                curves[(m, s)] = rows
    return curves


def fig_training(curves, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, scen in zip(axes, SCENARIO_ORDER):
        for (m, s), rows in sorted(curves.items()):
            x = [r["step"] for r in rows]
            y = [r.get(f"dice_{scen}", np.nan) for r in rows]
            ls = "-" if s == 0 else ("--" if s == 1 else ":")
            ax.plot(x, y, ls, color=SERIES[m], lw=2.0 if s == 0 else 1.4,
                    alpha=1.0 if s == 0 else 0.75, marker="o", ms=3.5,
                    label=f"{SERIES_LABEL[m]}" if s == 0 else None)
        ax.set_title(SCEN_LABEL[scen], color=INK)
        ax.set_xlabel("优化器更新次数")
        _strip(ax)
    axes[0].set_ylabel("验证集平均病例 Dice")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", fontsize=8)
    fig.suptitle("验证 Dice 随训练的变化（实线 = 种子 0；不同种子用线型区分）",
                 y=1.04, color=INK, fontsize=10, fontweight="bold")
    return finish(fig, out)


def fig_scenario_bars(summary, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    seeds = sorted({r["seed"] for r in summary})
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.36
    for k, m in enumerate(["A", "B"]):
        means, errs = [], []
        for scen in SCENARIO_ORDER:
            vals = [r["dice_mean"] for r in summary
                    if r["model"] == m and r["scenario"] == scen and r["seed"] == 0]
            stds = [r["dice_std"] for r in summary
                    if r["model"] == m and r["scenario"] == scen and r["seed"] == 0]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.mean(stds) if stds else 0.0)
        bars = ax.bar(x + (k - 0.5) * (width + 0.02), means, width,
                      color=SERIES[m], label=SERIES_LABEL[m],
                      yerr=errs, capsize=3, error_kw=dict(ecolor=INK2, lw=1.0))
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIO_ORDER], fontsize=8.5)
    ax.set_ylabel("测试集平均病例 Dice")
    ax.set_ylim(0, 1.12)          # 留出空间给柱顶数值标签
    ax.set_title(f"各场景 Dice（种子 0；误差棒为病例间标准差）", color=INK)
    # 图例放到坐标区外，避免与柱顶数值标签重叠
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2,
              fontsize=8.5, borderaxespad=0)
    _strip(ax)
    return finish(fig, out)


def fig_per_case(metrics, out, scenario=PRIMARY_SCENARIO, seed=0):
    a = {r["patient_id"]: float(r["dice"]) for r in metrics
         if r["model"] == "A" and r["scenario"] == scenario and int(r["seed"]) == seed}
    b = {r["patient_id"]: float(r["dice"]) for r in metrics
         if r["model"] == "B" and r["scenario"] == scenario and int(r["seed"]) == seed}
    ids = sorted(set(a) & set(b), key=lambda c: b[c] - a[c])
    if not ids:
        return None
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(ids) + 1.6))
    for i, cid in enumerate(ids):
        win = b[cid] > a[cid]
        ax.plot([a[cid], b[cid]], [i, i], color=(SERIES["B"] if win else SERIES["A"]),
                lw=1.6, alpha=0.55, zorder=1, solid_capstyle="round")
    ax.scatter([a[c] for c in ids], y, s=42, color=SERIES["A"], zorder=3,
               label=SERIES_LABEL["A"], edgecolor="white", linewidth=1.0)
    ax.scatter([b[c] for c in ids], y, s=42, color=SERIES["B"], zorder=3,
               label=SERIES_LABEL["B"], edgecolor="white", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("BraTS20_Training_", "") for c in ids], fontsize=7)
    ax.set_ylabel("测试病例编号")
    ax.set_xlabel("前景 Dice")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f"{SCEN_LABEL[scenario]} 逐病例配对比较（种子 {seed}）\n"
                 f"连线颜色 = 提升方向（橙=B 更好）", color=INK, fontsize=9.5)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    _strip(ax)
    ax.grid(axis="y", visible=False)
    return finish(fig, out)


def fig_diff_strip(metrics, out, seed=0):
    fig, axes = plt.subplots(len(SCENARIO_ORDER), 1, figsize=(7.2, 4.4), sharex=True)
    for ax, scen in zip(axes, SCENARIO_ORDER):
        a = {r["patient_id"]: float(r["dice"]) for r in metrics
             if r["model"] == "A" and r["scenario"] == scen and int(r["seed"]) == seed}
        b = {r["patient_id"]: float(r["dice"]) for r in metrics
             if r["model"] == "B" and r["scenario"] == scen and int(r["seed"]) == seed}
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        d = np.array([b[c] - a[c] for c in ids])
        boot = paired_bootstrap(d, 2000, seed=42)
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        ax.scatter(d, np.full_like(d, 1.0) + np.random.default_rng(0).uniform(-0.12, 0.12, d.size),
                   s=34, color=SERIES["B"], alpha=0.75, zorder=3, edgecolor="white", linewidth=0.8)
        ax.plot([boot["lo"], boot["hi"]], [0.45, 0.45], color=INK2, lw=2.0, zorder=3,
                solid_capstyle="butt")
        ax.plot([boot["mean"]], [0.45], "o", color=INK, ms=5.5, zorder=4)
        ax.set_ylim(0.25, 1.45)
        ax.set_yticks([])
        ax.set_title(f"{SCEN_LABEL[scen]}   Δ均值 {boot['mean']:+.3f}"
                     f"  95%区间 [{boot['lo']:+.3f}, {boot['hi']:+.3f}]",
                     fontsize=8.5, loc="left", color=INK)
        _strip(ax)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("逐病例 Dice 差值（B − A）")
    fig.suptitle("配对差值分布：每点为一个测试病例，横线为配对 bootstrap 95% 区间",
                 y=1.02, color=INK, fontsize=10, fontweight="bold")
    return finish(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    fig_dir = os.path.join(out, "figures")
    apply_style()
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = ["A", "B"]

    curves = read_val_curves(os.path.join(out, "runs"), methods, seeds)
    if curves:
        print("->", fig_training(curves, os.path.join(fig_dir, "fig_training_curves.png")))

    mp = os.path.join(out, "metrics_per_case.csv")
    if not os.path.exists(mp) and glob.glob(os.path.join(out, "metrics_per_case_seed*.csv")):
        from run_eval import rebuild_combined
        rebuild_combined(out)          # 由逐种子结果合并，不重跑推理
    if os.path.exists(mp):
        metrics = list(csv.DictReader(open(mp)))
        summary = list(csv.DictReader(open(os.path.join(out, "summary.csv"))))
        for r in summary:
            r["seed"] = int(r["seed"])
            r["dice_mean"] = float(r["dice_mean"]); r["dice_std"] = float(r["dice_std"])
        print("->", fig_scenario_bars(summary, os.path.join(fig_dir, "fig_dice_by_scenario.png")))
        print("->", fig_per_case(metrics, os.path.join(fig_dir, "fig_per_case_C1.png")))
        print("->", fig_diff_strip(metrics, os.path.join(fig_dir, "fig_diff_strip.png")))
    else:
        print("metrics_per_case.csv 尚未生成，跳过指标图。")


if __name__ == "__main__":
    main()
