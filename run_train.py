#!/usr/bin/env python3
"""阶段二：从实验目录读取任务配置，训练 A / B。

用法:
    python3 run_train.py --method A --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 必须在 import torch 之前设置：关闭缓存分配器的大块预留，避免长期训练中
# reserved 远超 allocated 而触发伪 OOM。
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import Config, save_config, load_config
from fmexp.train import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="覆盖学习率（调参敏感性实验用）")
    ap.add_argument("--tag", default="", help="运行目录后缀，用于区分调参实验")
    ap.add_argument("--val-every", type=int, default=None, help="验证间隔（默认 2000 次更新）")
    ap.add_argument("--val-mode", choices=["balanced", "full"], default="balanced",
                    help="balanced: 每例固定一个场景；full: 每例全部场景")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    cfg = load_config(os.path.join(out, "config_preprocess.yaml"))
    # 旧预处理配置可能保存旧间隔；训练协议以当前默认值/显式 CLI 为准。
    cfg.val_every = Config().val_every
    cfg.val_mode = args.val_mode
    if args.val_every is not None:
        cfg.val_every = args.val_every
    if args.steps:
        cfg.max_steps = args.steps
    if args.lr:
        cfg.lr = args.lr

    splits = json.load(open(os.path.join(out, "splits.json")))
    run_dir = os.path.join(out, "runs", f"{args.method}_seed{args.seed}{args.tag}")
    os.makedirs(run_dir, exist_ok=True)

    # 每个模型独立落盘一份配置（方案 §8）
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    save_config(cfg, os.path.join(out, f"config_{args.method}.yaml"))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"== 训练 方法{args.method} seed={args.seed} 设备={device} ==")
    train_one(args.method, cfg, args.seed, splits, os.path.join(out, "cache"), run_dir, device)


if __name__ == "__main__":
    main()
