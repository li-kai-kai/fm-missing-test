#!/usr/bin/env python3
"""阶段 0：数据审计 + 病例选择 + 患者级划分 + 预处理缓存。

对应方案 §2.1 / §2.2 / §2.3 与 §8 的产物要求。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import Config, MODALITIES, SCENARIOS, save_config
from fmexp.data import (LABEL_MAPPING, all_labeled_cases, case_id, load_case,
                        make_splits, preprocess_case, select_cases)


def audit(data_root: str, out_dir: str) -> dict:
    ids = all_labeled_cases(data_root)
    shapes, zooms, label_vals = Counter(), Counter(), Counter()
    import nibabel as nib
    for cid in ids:
        seg = nib.load(os.path.join(data_root, cid, f"{cid}_seg.nii.gz"))
        shapes[seg.shape] += 1
        zooms[tuple(round(float(z), 4) for z in seg.header.get_zooms())] += 1
        for u, c in zip(*np.unique(np.asanyarray(seg.dataobj), return_counts=True)):
            label_vals[float(u)] += int(c)
    a = {
        "data_root": data_root,
        "n_labeled_cases": len(ids),
        "shapes": {str(k): v for k, v in shapes.items()},
        "zooms": {str(k): v for k, v in zooms.items()},
        "label_values_global": {str(k): v for k, v in sorted(label_vals.items())},
        "already_registered_and_isotropic": True,
        "resampling_needed": False,
    }
    with open(os.path.join(out_dir, "data_audit.json"), "w") as f:
        json.dump(a, f, indent=2, ensure_ascii=False)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--force", action="store_true", help="重新预处理已缓存的病例")
    args = ap.parse_args()

    cfg = Config(out_root=args.out)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    cache = os.path.join(out, "cache")
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)

    print("== 数据审计 ==")
    a = audit(cfg.data_root, out)
    print(json.dumps(a, indent=2, ensure_ascii=False))

    # 标签映射表落盘（方案 §2.2）
    with open(os.path.join(out, "label_mapping.json"), "w") as f:
        json.dump(LABEL_MAPPING, f, indent=2, ensure_ascii=False)

    print("\n== 选择病例与划分 ==")
    cases = select_cases(cfg.data_root, cfg.n_cases, cfg.split_seed)
    splits = make_splits(cases, cfg.n_train, cfg.n_val, cfg.n_test, cfg.split_seed)
    splits_doc = {
        "split_seed": cfg.split_seed,
        "unit": "patient（BraTS 每病例即一位患者，患者级划分天然成立）",
        "source_pool": f"BraTS2020 有标注训练病例 {len(all_labeled_cases(cfg.data_root))} 例中按种子 {cfg.split_seed} 随机选 {cfg.n_cases} 例",
        "n": {k: len(v) for k, v in splits.items()},
        **splits,
    }
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(splits_doc, f, indent=2, ensure_ascii=False)
    print({k: len(v) for k, v in splits.items()})

    print("\n== 预处理（完整四模态，归一化后不置零；置零在采样/推理时按场景进行）==")
    metas = []
    for i, cid in enumerate(cases):
        m = preprocess_case(cfg.data_root, cid, cache, reorient_to=cfg.reorient_to,
                            overwrite=args.force)
        metas.append(m)
        if (i + 1) % 10 == 0 or i == len(cases) - 1:
            print(f"  {i+1}/{len(cases)} 完成，最近: {cid} 肿瘤体素={m['tumor_voxels']}")

    empty = [m["case_id"] for m in metas if m["tumor_voxels"] == 0]
    if empty:
        print(f"!! 警告：以下病例肿瘤体素为 0: {empty}")
    else:
        print("全部病例均含肿瘤体素。")

    save_config(Config(**{**cfg.__dict__}), os.path.join(out, "config_preprocess.yaml"))
    print("\n完成。产物 ->", out)


if __name__ == "__main__":
    main()
