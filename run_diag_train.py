#!/usr/bin/env python3
"""诊断用重训入口。复用 train_one，产物只写进 <out>/diagnostics/runs/。

三个实验：
  tiny-fit  2 个训练病例、关增强、patch 全部围绕肿瘤，3000 步 —— 检验能否拟合
  bgzero    保持背景严格为 0 的增强变体，10k 步 —— 检验背景分布错位的影响
  long      延长训练到 40k 步（B seed0）—— 检验是欠训练还是不稳定

用法:
    python3 run_diag_train.py --out experiment_multiclass --exp tiny-fit
    python3 run_diag_train.py --out experiment_multiclass --exp bgzero
    python3 run_diag_train.py --out experiment_multiclass --exp long
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import load_config, save_config
from fmexp.train import train_one

# 每个实验的配置覆盖与说明
EXPERIMENTS = {
    "tiny-fit": {
        "methods": ("A", "B"), "seed": 0, "splits": "tiny",
        "overrides": {"max_steps": 3000, "augment": False, "tumor_center_prob": 1.0,
                      "val_every": 10 ** 9},
        "desc": "2 个训练病例、关闭增强、patch 全部围绕肿瘤；检验能否拟合（不参与性能比较）",
    },
    "bgzero": {
        "methods": ("A", "B"), "seed": 0, "splits": "full",
        "overrides": {"aug_keep_background_zero": True, "val_every": 5000},
        "desc": "增强只在脑支撑内施加、背景严格为 0；对照原 B_seed0（val_every=2000）",
    },
    "long": {
        "methods": ("B",), "seed": 0, "splits": "full",
        "overrides": {"max_steps": 40000, "val_every": 5000},
        "desc": "延长训练到 40k 步。前 10k 步与 B_seed0 使用同一 RNG 流，应逐位复现",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment_multiclass")
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="覆盖该实验的 max_steps")
    ap.add_argument("--tumor-center-prob", type=float, default=None,
                    help="覆盖 patch 采样：1.0=全部围绕肿瘤，0.5=默认（覆盖不同脑区）")
    ap.add_argument("--tag", default="", help="运行目录后缀，避免覆盖同名的诊断重训")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    if not os.path.exists(os.path.join(out, "config_preprocess.yaml")):
        raise SystemExit(f"找不到 {out}/config_preprocess.yaml，请确认 --out 指向四分类实验目录")

    spec = EXPERIMENTS[args.exp]
    seed = args.seed if args.seed is not None else spec["seed"]
    splits = json.load(open(os.path.join(out, "splits.json")))
    if spec["splits"] == "tiny":
        # validation_assignments 要求验证病例数不少于场景数（4），故给 4 例。
        tiny_tr = splits["train"][:2]
        splits = {"train": tiny_tr, "val": splits["val"][:4]}
    overrides = dict(spec["overrides"])
    if args.steps:
        overrides["max_steps"] = args.steps
    if args.tumor_center_prob is not None:
        overrides["tumor_center_prob"] = args.tumor_center_prob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== 诊断重训 {args.exp} seed={seed} 设备={device} ==")
    print(f"   说明：{spec['desc']}")
    print(f"   病例：train={len(splits['train'])} val={len(splits['val'])} "
          f"覆盖={overrides}")

    for method in spec["methods"]:
        cfg = load_config(os.path.join(out, "config_preprocess.yaml"))
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise SystemExit(f"配置项不存在: {k}")
            setattr(cfg, k, v)
        run_dir = os.path.join(out, "diagnostics", "runs",
                               f"{args.exp}_{method}_seed{seed}{args.tag}")
        os.makedirs(run_dir, exist_ok=True)
        # 只写进诊断目录；绝不调用 run_train.py 那种写实验根 config_{A,B}.yaml 的路径
        save_config(cfg, os.path.join(run_dir, "config.yaml"))
        with open(os.path.join(run_dir, "diag_meta.json"), "w") as f:
            json.dump({"experiment": args.exp, "method": method, "seed": seed,
                       "desc": spec["desc"], "overrides": overrides,
                       "train_cases": splits["train"], "val_cases": splits["val"]},
                      f, indent=2, ensure_ascii=False)
        train_one(method, cfg, seed, splits, os.path.join(out, "cache"), run_dir, device)


if __name__ == "__main__":
    main()
