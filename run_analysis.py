#!/usr/bin/env python3
"""失败案例分析（方案 §7.4：「改善只来自少数病例」/「先分析病灶大小和失败案例」）。

按真实肿瘤体积分箱，看逐病例 Dice 差值 B−A 是否与病灶大小相关；并单独列出
完全漏检与 FM 相对 A 的极端改善/退步病例。不做任何阈值式的“成功”判定。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import PRIMARY_SCENARIO, SCENARIO_ORDER
from run_eval import coerce_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default=PRIMARY_SCENARIO)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    rows = [coerce_row(r) for r in csv.DictReader(open(os.path.join(out, "metrics_per_case.csv")))]

    # 真实肿瘤体积（体素）来自预处理元信息
    vol = {}
    for p in glob.glob(os.path.join(out, "cache", "*.json")):
        m = json.load(open(p))
        vol[m["case_id"]] = m["tumor_voxels"]

    a = {r["patient_id"]: r for r in rows
         if r["model"] == "A" and r["scenario"] == args.scenario and r["seed"] == args.seed}
    b = {r["patient_id"]: r for r in rows
         if r["model"] == "B" and r["scenario"] == args.scenario and r["seed"] == args.seed}
    ids = [c for c in sorted(set(a) & set(b)) if c in vol]   # 丢弃查不到体积的病例
    if not ids:
        print("没有可用结果（或 cache 元信息缺失）"); return

    print(f"\n== 场景 {args.scenario}，种子 {args.seed}：按真实肿瘤体积分箱 ==")
    sizes = np.array([vol.get(c, np.nan) for c in ids], dtype=float)
    diffs = np.array([b[c]["dice"] - a[c]["dice"] for c in ids])
    order = np.argsort(sizes)
    nbin = 4
    print(f"{'分箱(真实体素)':>24} {'例数':>5} {'A 均值':>9} {'B 均值':>9} {'B−A':>9}")
    bin_report = []
    for k in range(nbin):
        idx = order[k * len(ids) // nbin:(k + 1) * len(ids) // nbin]
        if len(idx) == 0:
            continue
        sub = [ids[i] for i in idx]
        am = float(np.mean([a[c]["dice"] for c in sub]))
        bm = float(np.mean([b[c]["dice"] for c in sub]))
        lo, hi = int(sizes[idx].min()), int(sizes[idx].max())
        print(f"{f'{lo}–{hi}':>24} {len(sub):>5} {am:>9.4f} {bm:>9.4f} {bm-am:>+9.4f}")
        bin_report.append({"lo": lo, "hi": hi, "n": len(sub), "A": am, "B": bm, "diff": bm - am})

    if np.std(sizes) > 0 and np.std(diffs) > 0:
        r = float(np.corrcoef(sizes, diffs)[0, 1])
        print(f"\n肿瘤体积与 Dice 差值 B−A 的相关系数 r = {r:+.3f}（{len(ids)} 例）")
    else:
        r = float("nan")

    print("\n== 完全漏检 ==")
    miss_a = [c for c in ids if a[c]["complete_miss"]]
    miss_b = [c for c in ids if b[c]["complete_miss"]]
    print(f"A: {len(miss_a)} 例 {[c.replace('BraTS20_Training_','') for c in miss_a]}")
    print(f"B: {len(miss_b)} 例 {[c.replace('BraTS20_Training_','') for c in miss_b]}")

    print("\n== 极端病例（按结果选出，不是随机抽样）==")
    srt = sorted(ids, key=lambda c: b[c]["dice"] - a[c]["dice"])
    for tag, cid in (("B 最大退步", srt[0]), ("B 最大改善", srt[-1])):
        print(f"{tag}: {cid.replace('BraTS20_Training_','')} "
              f"A={a[cid]['dice']:.4f} B={b[cid]['dice']:.4f} "
              f"Δ={b[cid]['dice']-a[cid]['dice']:+.4f} "
              f"真实体素={vol.get(cid)} HD95 A={a[cid]['hd95']:.1f} B={b[cid]['hd95']:.1f}")

    res = {"scenario": args.scenario, "seed": args.seed, "bins": bin_report,
           "corr_size_diff": r,
           "complete_miss_A": miss_a, "complete_miss_B": miss_b,
           "worst_regression": srt[0], "best_improvement": srt[-1],
           "per_case_diff": {c: float(b[c]["dice"] - a[c]["dice"]) for c in ids}}
    path = os.path.join(out, f"analysis_{args.scenario}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("->", path)


if __name__ == "__main__":
    main()
