"""绘图工具。

配色取自已通过校验的分类调色板（validate_palette.js，light 模式、全配对）：
  A（普通分割）= 蓝 #2a78d6，B（FM 分割）= 橙 #eb6834，第三槽位 青绿 #1baf7a。
三槽位全配对通过 CVD 与常视分离度门槛；青绿对浅色底的对比度低于 3:1，
因此凡使用青绿处一律附可见的文字标签（relief 规则）。
序列色用单一蓝色阶，发散色用蓝<->红。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = {"A": "#2a78d6", "B": "#eb6834"}          # 按实体固定，不随排名变化
SERIES_LABEL = {"A": "A 直接分割", "B": "B FM 生成"}
ACCENT = "#1baf7a"
CRITICAL = "#d03b3b"

SCEN_LABEL = {"C0": "C0 完整模态", "C1": "C1 缺 T1ce（主要）", "C2": "C2 T2+FLAIR"}


_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def _register_cjk():
    """注册中文字体，否则图上的中文会渲染成方框。"""
    import matplotlib.font_manager as fm
    for p in _CJK_CANDIDATES:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                continue
    fams = {f.name for f in fm.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC"):
        if name in fams:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": _register_cjk(),
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })


def finish(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _strip(ax, top_right=True):
    ax.spines["top"].set_visible(not top_right)
    ax.spines["right"].set_visible(not top_right)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- 医学图像

def best_slice(vol_or_mask, axis=2, margin=0.15):
    """取包含最多前景的切片（仅用于选展示层，不参与任何指标计算）。"""
    counts = vol_or_mask.sum(axis=tuple(i for i in range(3) if i != axis))
    nz = np.flatnonzero(counts)
    if nz.size == 0:
        return vol_or_mask.shape[axis] // 2
    lo, hi = nz[0], nz[-1]
    cut = lo + margin * (hi - lo)
    cand = nz[nz >= cut]
    return int(cand[0]) if cand.size else int(nz[nz.argmax()])


def draw_slice(ax, bg, overlays=(), title="", axis=2, idx=None):
    """bg: [D,H,W] 灰度底图；overlays: [(mask, color, label)]，按 contour 画轮廓。"""
    idx = best_slice(bg if np.any(bg) else overlays[0][0], axis) if idx is None else idx
    sl = [slice(None)] * 3
    sl[axis] = idx
    img = np.rot90(bg[tuple(sl)])
    ax.imshow(img, cmap="gray", interpolation="nearest")
    hold = np.rot90(np.zeros_like(img))
    for mask, color, label in overlays:
        m = np.rot90(mask[tuple(sl)].astype(float))
        if m.max() > 0:
            ax.contour(m, levels=[0.5], colors=[color], linewidths=1.4, linestyles="solid")
            hold = np.maximum(hold, m)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(AXIS); s.set_linewidth(0.6)
