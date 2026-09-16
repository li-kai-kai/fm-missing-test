#!/usr/bin/env python3
"""方案 §6 阶段二规定的排查：FM 验证结果差时，在**相同 checkpoint 和相同初始噪声**上
比较 16 步与 32 步 Euler，只在验证集上做这个判断。

同时给出每个场景的平均病例 Dice，便于判断“生成步数不足”是否是主因。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import Config, SCENARIO_ORDER
from fmexp.data import VolumeStore, avail_vector
from fmexp.infer import init_noise_for_case, predict_volume_B
from fmexp.metrics import dice_foreground
from fmexp.train import load_checkpoint
from fmexp.unet import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--steps", default="16,32")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.load(open(os.path.join(out, "splits.json")))
    cases = splits[args.split]

    ck = load_checkpoint(os.path.join(out, "runs", f"B_seed{args.seed}"), args.ckpt)
    model = build_model("B", cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"载入 B_seed{args.seed}/{args.ckpt} step={ck['step']} val_score={ck['score']:.4f}")
    store = VolumeStore(os.path.join(out, "cache"), cases)

    step_list = [int(s) for s in args.steps.split(",")]
    res = {s: {scen: [] for scen in SCENARIO_ORDER} for s in step_list}
    for cid in cases:
        case = store[cid]
        gt = case.seg
        mri_full = torch.from_numpy(case.img).to(device)
        # 同一步数的比较必须共用同一份初始噪声（方案 §5.2 / §6）
        noise = init_noise_for_case(cid, gt.shape, cfg.eval_seed, device)
        for scen in SCENARIO_ORDER:
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)
            for s in step_list:
                pred = predict_volume_B(model, mri, av, cfg, noise, steps=s)
                res[s][scen].append(dice_foreground(pred, gt))
        print(f"  {cid} 完成", flush=True)

    print(f"\n{'步数':>6} " + " ".join(f"{s:>10}" for s in SCENARIO_ORDER) + f"{'三场景均值':>12}")
    summary = {}
    for s in step_list:
        means = [float(np.mean(res[s][scen])) for scen in SCENARIO_ORDER]
        summary[s] = {"per_scenario": dict(zip(SCENARIO_ORDER, means)),
                      "mean": float(np.mean(means))}
        print(f"{s:>6} " + " ".join(f"{m:>10.4f}" for m in means) + f"{np.mean(means):>12.4f}")

    path = os.path.join(out, f"fm_steps_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump({"seed": args.seed, "ckpt": args.ckpt, "split": args.split,
                   "steps": step_list, "result": summary,
                   "per_case": {str(s): {k: [float(x) for x in v] for k, v in res[s].items()}
                                for s in step_list}},
                  f, indent=2, ensure_ascii=False)
    print("->", path)


if __name__ == "__main__":
    main()
