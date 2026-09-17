#!/usr/bin/env python3
"""真实训练病例上的四分类短程调通；输出不用于方法效果比较。"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from fmexp.config import Config, save_config
from fmexp.data import VolumeStore, make_batch, preprocess_case
from fmexp.train import _to_device, forward_loss_A, forward_loss_B, validate
from fmexp.unet import build_model, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_dryrun/multiclass")
    ap.add_argument("--splits-from", default="experiment/splits.json")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    if args.steps <= 0:
        ap.error("--steps 必须为正数")
    cfg = Config(task="multiclass_missing_one", max_steps=args.steps)
    out = Path(args.out)
    cache = out / "cache"
    cases = json.loads(Path(args.splits_from).read_text())["train"][:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    for cid in cases:
        preprocess_case(cfg.data_root, cid, str(cache), task=cfg.task)
        print(f"预处理完成 {cid}", flush=True)
    store = VolumeStore(str(cache), cases, cfg.task)
    save_config(cfg, str(out / "config.yaml"))
    report = {"purpose": "smoke_only_not_performance_evaluation", "cases": cases,
              "steps": args.steps, "device": str(device), "methods": {}}
    for method in ("A", "B"):
        torch.manual_seed(0)
        model = build_model(method, cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        losses = []
        for step in range(args.steps):
            batch = make_batch(store, cases, 0, step, cfg.patch, cfg.tumor_center_prob,
                               cfg.batch_size, n_classes=cfg.n_classes, scenario_order=cfg.scenarios)
            assert np.all(batch["avail"].sum(1) == 3)
            assert np.all(batch["y1"].sum(1) == 1)
            b = _to_device(batch, device)
            if method == "A":
                loss, _ = forward_loss_A(model, b["mri"], b["avail"], b["y1"], cfg)
            else:
                loss, _ = forward_loss_B(model, b["mri"], b["avail"], b["y1"], 0, step, cfg, device)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失非有限数")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12., error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        print(f"{method}: {args.steps} 次更新完成，loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
        start = time.perf_counter()
        rows = validate(model, method, cfg, store, cases[:1], device)
        seconds = time.perf_counter() - start
        assert len(rows) == 12
        assert all(np.isfinite(r["dice"]) for r in rows)
        report["methods"][method] = {"params": n_params(model), "losses": losses,
                                      "full_volume_seconds_4_scenarios": seconds, "rows": rows}
        print(f"{method}: 全体积×四种缺失模式完成，WT/TC/ET 共 12 行，耗时 {seconds:.2f}s", flush=True)
        del model, opt, b, loss
    path = out / "smoke_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"短程检查通过；不是正式实验结果：{path}", flush=True)


if __name__ == "__main__":
    main()
