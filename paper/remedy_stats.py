#!/usr/bin/env python3
"""修复臂 vs 基线 B 的逐病例配对比较。

沿用主结果的口径（run_eval.paired_diffs）：在每个 (训练种子, 缺失场景, 区域)
单元内，按病人配对求差值再 bootstrap，2000 次重采样，区间 95%。只覆盖评价
噪声，不覆盖训练随机性；12 个单元未做多重比较校正。
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))
from fm_missing_mri_test.fmexp.metrics import paired_bootstrap  # noqa: E402

EV = os.path.join(ROOT, "experiment_multiclass/evaluation/test")
ARMS = [("_fgw9", "foreground weighting"),
        ("_tg1", "late-time weighting"),
        ("_fgw9tg1", "foreground + late-time")]


def load(path, model=None):
    rows = list(csv.DictReader(open(path)))
    if model:
        rows = [r for r in rows if r["model"] == model]
    return {(r["patient_id"], r["seed"], r["scenario"], r["region"]): float(r["dice"])
            for r in rows}


def main():
    base = load(os.path.join(EV, "main/metrics_per_case.csv"), "B")
    for tag, label in ARMS:
        p = os.path.join(EV, tag, "metrics_per_case.csv")
        if not os.path.exists(p):
            print(f"{label}: 尚无评估结果")
            continue
        arm = load(p)
        common = sorted(set(base) & set(arm))
        print(f"\n=== {label}  −  基线 B ({len(common)} 个配对单元) ===")
        cells = defaultdict(list)
        for c in common:
            cells[(c[1], c[2], c[3])].append(arm[c] - base[c])

        for region in ("WT", "TC", "ET"):
            rows = [(k, paired_bootstrap(np.array(v))) for k, v in sorted(cells.items())
                    if k[2] == region]
            if not rows:
                continue
            means = [r["mean"] for _, r in rows]
            pos = sum(1 for _, r in rows if r["lo"] > 0)
            neg = sum(1 for _, r in rows if r["hi"] < 0)
            print(f"  {region}: 均值差 {np.mean(means):+.4f} "
                  f"(各单元范围 {min(means):+.3f} … {max(means):+.3f})；"
                  f"区间全为正 {pos}/{len(rows)}，全为负 {neg}/{len(rows)}")
            for k, r in rows:
                print(f"      seed{k[0]} {k[1]:<14} diff={r['mean']:+.4f} "
                      f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")


if __name__ == "__main__":
    main()
