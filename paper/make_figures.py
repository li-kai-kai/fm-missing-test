#!/usr/bin/env python3
"""生成论文插图（PDF）。只读诊断产物与已有预测，不修改任何实验数据。

配色取自校验过的分类调色板（validate_palette.js，light 模式全项通过）：
  #2a78d6 蓝 / #eb6834 橙 / #1baf7a 青 / #eda100 黄
对比度 WARN 的满足方式是「可见标签 + 正文表格」；同时每条曲线用不同线型
做二次编码，黑白打印和色觉障碍下仍可区分。
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "experiment_multiclass/diagnostics")
OUT = os.path.join(ROOT, "paper/figures")

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LS = ["-", "--", "-.", ":"]
MARK = ["o", "s", "^", "D"]
GRID = dict(color="0.85", linewidth=0.5)
REF = "0.45"   # 参考线用灰，不算一个系列

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _clean(ax):
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)


def fig_mechanism():
    """机制图：场在 L2 意义上准；误差随 t 上升；背景体素主导聚合误差。"""
    d = json.load(open(os.path.join(DIAG, "time_B_seed0.json")))
    bins = d["velocity_mse_by_t"]["bins"]
    t = np.array([b["t"] for b in bins])
    mse = np.array([b["mse_model"] for b in bins])
    ref = np.array([b["mse_prior_only_ref"] for b in bins])
    var = np.array([b["var_v"] for b in bins])
    r2 = np.array([b["r2"] for b in bins])
    cls = {c: np.array([b["per_class"][str(c)]["mse"] for b in bins]) for c in range(4)}
    # 背景体素占比，用于说明「聚合值被谁撑起来」
    n_tot = np.array([sum(b["per_class"][str(c)]["n"] for c in range(4)) for b in bins])
    bg_frac = np.array([b["per_class"]["0"]["n"] / n for b, n in zip(bins, n_tot)])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))

    ax = axes[0]
    ax.plot(t, mse, LS[0], color=C[0], marker=MARK[0], ms=3, lw=1.4,
            label="model velocity MSE")
    # 先验参考在高 t 段趋于 0（此时 y_t 已几乎决定答案），该段无意义，裁掉
    keep = ref > 5e-3
    ax.plot(t[keep], ref[keep], LS[1], color=C[1], marker=MARK[1], ms=3, lw=1.4,
            label="class-prior reference")
    ax.axhline(float(var.mean()), color=REF, ls="--", lw=1.1,
               label=r"$\mathrm{Var}(v)$")
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 3)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("velocity MSE")
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    _clean(ax)
    ax.set_title("(a) The field is accurate in $L_2$", loc="left")
    ax.annotate(rf"$R^2$ {r2[0]:.3f} $\rightarrow$ {r2[-1]:.3f}",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=6.5, color="0.25")

    ax = axes[1]
    names = ["background", "NCR/NET", "ED", "ET"]
    for c in range(4):
        ax.plot(t, cls[c], LS[c], color=C[c], marker=MARK[c], ms=3, lw=1.4,
                label=names[c])
    ax.set_yscale("log")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("velocity MSE by class")
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.6)
    _clean(ax)
    ax.set_title(rf"(b) Background dominates ({bg_frac.mean()*100:.0f}% of voxels)",
                 loc="left")

    fig.savefig(os.path.join(OUT, "fig_mechanism.pdf"))
    plt.close(fig)


def fig_sweep():
    """求解器越准结果越差；端点置信间隔与步数无关。"""
    s = json.load(open(os.path.join(DIAG, "steps_B_seed0.json")))
    per = s["per_scenario"]["missing_t1ce"]["summary"]
    steps = sorted(int(k) for k in per)
    regions = ["WT", "TC", "ET"]

    e = json.load(open(os.path.join(DIAG, "endpoint_B_seed0.json")))
    es = e["summary"]
    e_steps = sorted(int(k) for k in es)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))

    ax = axes[0]
    for i, r in enumerate(regions):
        y = [per[str(k)][r] for k in steps]
        ax.plot(steps, y, LS[i], color=C[i], marker=MARK[i], ms=3, lw=1.4, label=r)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.set_xticklabels([str(k) for k in steps])
    ax.set_xlabel("Euler steps")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, loc="upper right")
    _clean(ax)
    ax.set_title("(a) More accurate integration is worse", loc="left")

    ax = axes[1]
    corr = [es[str(k)]["margin_correct_mean"] for k in e_steps]
    wrong = [es[str(k)]["margin_wrong_mean"] for k in e_steps]
    med = [es[str(k)]["margin_median"] for k in e_steps]
    x = np.arange(len(e_steps))
    ax.bar(x - 0.19, corr, width=0.36, color=C[0], edgecolor="white",
           linewidth=0.6, label="correct voxels")
    ax.bar(x + 0.19, wrong, width=0.36, color=C[1], edgecolor="white",
           linewidth=0.6, label="wrong voxels")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in e_steps])
    ax.set_xlabel("Euler steps")
    ax.set_ylabel("mean endpoint margin")
    ax.set_ylim(0, 1.15)
    _clean(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4)
    ax.annotate(rf"median margin {med[0]:.3f} $\rightarrow$ {med[-1]:.3f}; "
                rf"only {es['16']['frac_margin_lt_0.25']*100:.1f}% of voxels "
                rf"below $0.25$",
                xy=(0.5, -0.34), xycoords="axes fraction", ha="center",
                fontsize=6, color="0.35")
    ax.set_title("(b) Errors are not near-ties, and steps do not help", loc="left")

    fig.savefig(os.path.join(OUT, "fig_sweep.pdf"))
    plt.close(fig)


def fig_qualitative():
    """定性对比：同一病例同一缺失场景下 A 与 B 的预测。"""
    import nibabel as nib
    pred_root = os.path.join(ROOT, "experiment_multiclass/predictions/test")
    scen = "missing_flair"
    case = None
    for cid in sorted(os.listdir(pred_root)):
        fa = os.path.join(pred_root, cid, f"A_seed0_{scen}.nii.gz")
        fb = os.path.join(pred_root, cid, f"B_seed0_{scen}.nii.gz")
        if os.path.exists(fa) and os.path.exists(fb):
            case = cid
            break
    if case is None:
        print("找不到可用的预测，跳过定性图")
        return

    a = np.asarray(nib.load(os.path.join(pred_root, case, f"A_seed0_{scen}.nii.gz")).dataobj)
    b = np.asarray(nib.load(os.path.join(pred_root, case, f"B_seed0_{scen}.nii.gz")).dataobj)
    gt = np.load(os.path.join(ROOT, "experiment_multiclass/cache", f"{case}.npz"))["seg"]

    # 取肿瘤面积最大的轴位层
    counts = (gt > 0).sum(axis=(0, 1))
    z = int(np.argmax(counts))

    # 先转到显示用的坐标系，再算外接框——两者坐标系不同，混用会让裁剪框错位
    def disp(v):
        return np.rot90(v[:, :, z].T)

    dg, da, db = disp(gt), disp(a), disp(b)
    fg = (dg > 0) | (da > 0) | (db > 0)
    ys, xs = np.nonzero(fg)
    pad = 14
    y0, y1 = max(0, ys.min() - pad), min(fg.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(fg.shape[1], xs.max() + pad)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    # 原始标签值 0/1/2/4，用 vmin/vmax 映射到 0..3 号色位
    cmap = ListedColormap(["#f4f4f2", C[2], C[1], C[3]])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    for ax, vol, title in ((axes[0], dg, "ground truth"),
                           (axes[1], da, "A: discriminative"),
                           (axes[2], db, "B: flow matching")):
        ax.imshow(vol, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(title, loc="left")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)   # imshow 的 y 轴朝下，外接框要反着给
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [Patch(facecolor=C[2], label="NCR/NET"),
               Patch(facecolor=C[1], label="ED"),
               Patch(facecolor=C[3], label="ET")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{case}, {scen.replace('_', ' ')}, axial slice {z}",
                 fontsize=7.5, y=1.0)
    fig.savefig(os.path.join(OUT, "fig_qualitative.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_mechanism()
    fig_sweep()
    fig_qualitative()
    print("已生成:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
