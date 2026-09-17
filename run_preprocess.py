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
from fmexp.config import Config, MODALITIES, SCENARIOS, TASKS, save_config, load_config
from fmexp.data import (LABEL_MAPPING, all_labeled_cases, case_id, load_case,
                        make_splits, preprocess_case, select_cases, MULTICLASS_LABEL_MAPPING)


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", choices=TASKS, default="binary_wt")
    ap.add_argument("--splits-from", help="复用已有患者划分 JSON；保持与旧实验一致")
    ap.add_argument("--force", action="store_true", help="重新预处理已缓存的病例")
    args = ap.parse_args()

    cfg = Config(task=args.task, out_root=args.out or (
        "experiment_multiclass" if args.task == "multiclass_missing_one" else "experiment"))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.out_root)
    config_path = os.path.join(out, "config_preprocess.yaml")
    if os.path.exists(config_path) and load_config(config_path).task != cfg.task:
        raise ValueError("此目录已有其他任务，请为新实验使用独立 --out")
    cache = os.path.join(out, "cache")
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)

    print("== 数据审计 ==")
    a = audit(cfg.data_root, out)
    print(json.dumps(a, indent=2, ensure_ascii=False))

    # 标签映射表落盘（方案 §2.2）
    with open(os.path.join(out, "label_mapping.json"), "w") as f:
        json.dump(MULTICLASS_LABEL_MAPPING if cfg.n_classes == 4 else LABEL_MAPPING,
                  f, indent=2, ensure_ascii=False)

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
    if args.splits_from:
        with open(args.splits_from) as f:
            splits_doc = json.load(f)
        splits = {k: splits_doc[k] for k in ("train", "val", "test")}
        cases = [c for group in splits.values() for c in group]
        if len(set(cases)) != len(cases):
            raise ValueError("患者划分存在重复或交叉")
        if not set(cases) <= set(all_labeled_cases(cfg.data_root)):
            raise ValueError("划分包含不存在的病例")
        cfg.n_train, cfg.n_val, cfg.n_test = (len(splits[k]) for k in ("train", "val", "test"))
        cfg.n_cases = len(cases)
        cfg.split_seed = splits_doc.get("split_seed", cfg.split_seed)
        splits_doc["n"] = {k: len(v) for k, v in splits.items()}
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(splits_doc, f, indent=2, ensure_ascii=False)
    print({k: len(v) for k, v in splits.items()})

    print("\n== 预处理（完整四模态，归一化后不置零；置零在采样/推理时按场景进行）==")
    metas = []
    for i, cid in enumerate(cases):
        m = preprocess_case(cfg.data_root, cid, cache, reorient_to=cfg.reorient_to,
                            overwrite=args.force, task=cfg.task)
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
