"""训练与验证（方案 §4 / §6 阶段二）。

checkpoint 选择规则（两模型一致）：验证集三个场景平均病例 Dice 最大的那个。
FM 的验证必须走完整生成过程，不能只监控速度 MSE。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from .config import Config, SCENARIO_ORDER
from .data import VolumeStore, avail_vector, fm_noise, make_batch
from .infer import init_noise_for_case, predict_volume_A, predict_volume_B
from .losses import loss_A, loss_B
from .metrics import case_metrics
from .unet import build_model, n_params

__all__ = ["VolumeStore", "validate", "train_one", "load_checkpoint"]


def _to_device(batch: dict, device) -> dict:
    return {
        "mri": torch.from_numpy(batch["mri"]).to(device),
        "y1": torch.from_numpy(batch["y1"]).to(device),
        "avail": torch.from_numpy(batch["avail"]).to(device),
        "scenarios": batch["scenarios"],
    }


def _expand_avail_t(avail: torch.Tensor, shape) -> torch.Tensor:
    return avail.view(avail.shape[0], 4, 1, 1, 1).expand(avail.shape[0], 4, *shape[2:])


def forward_loss_A(model, mri, avail, y1, cfg):
    x = torch.cat([mri, _expand_avail_t(avail, mri.shape)], dim=1)
    logits = model(x)
    return loss_A(logits, y1)


def forward_loss_B(model, mri, avail, y1, seed, step, cfg, device):
    y0, t = fm_noise(seed, step, mri.shape[0], mri.shape[2:], device, torch.float32)
    yt = (1 - t) * y0 + t * y1
    target_v = y1 - y0
    tc = t.expand(t.shape[0], 1, *mri.shape[2:])
    x = torch.cat([mri, _expand_avail_t(avail, mri.shape), yt, tc], dim=1)
    pred_v = model(x)
    return loss_B(pred_v, target_v)


@torch.no_grad()
def validate(model, method: str, cfg: Config, store: VolumeStore,
             val_cases: Sequence[str], device) -> List[dict]:
    """验证/测试：覆盖完整脑体积，不做真实肿瘤位置引导的裁剪或 patch 选择。"""
    model.eval()
    rows = []
    for cid in val_cases:
        case = store[cid]
        mri_full = torch.from_numpy(case.img).to(device)   # [4,D,H,W]
        gt = case.seg
        noise = None
        if method == "B":
            noise = init_noise_for_case(cid, gt.shape, cfg.eval_seed, device)
        for scen in SCENARIO_ORDER:
            # 训练期选择 checkpoint 只按方案 §4.3 规定的平均病例 Dice，
            # 因此跳过 HD95（那一步的全脑距离变换是评价里最贵的一环）。
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)              # 归一化之后置零被隐藏模态
            t0 = time.time()
            if method == "A":
                prob = predict_volume_A(model, mri, av, cfg)
                pred = (prob > 0.5).astype(np.uint8)
            else:
                pred = predict_volume_B(model, mri, av, cfg, noise)
            dt = time.time() - t0
            m = case_metrics(pred, gt, fast=True)
            m.update({"case_id": cid, "scenario": scen, "seconds": dt})
            rows.append(m)
    model.train()
    return rows


def _score(rows: Sequence[dict]) -> float:
    per = {}
    for r in rows:
        per.setdefault(r["scenario"], []).append(r["dice"])
    return float(np.mean([np.mean(per[s]) for s in SCENARIO_ORDER]))


def train_one(method: str, cfg: Config, seed: int, splits: dict,
              cache_dir: str, run_dir: str, device, log=print) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    store = VolumeStore(cache_dir, splits["train"] + splits["val"])
    model = build_model(method, cfg).to(device)
    n_par = n_params(model)
    log(f"[{method} seed={seed}] 参数量 = {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path = os.path.join(run_dir, "log.jsonl")
    fh = open(log_path, "w")
    meta = {
        "method": method, "seed": seed, "params": n_par,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
        "patch": cfg.patch, "max_steps": cfg.max_steps, "val_every": cfg.val_every,
        "fm_steps": cfg.fm_steps, "sw_window": cfg.sw_window, "sw_overlap": cfg.sw_overlap,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "train_cases": splits["train"], "val_cases": splits["val"],
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    best_score, best_step = -np.inf, -1
    t_start = time.time()
    model.train()

    for step in range(cfg.max_steps + 1):
        if step % cfg.val_every == 0 or step == cfg.max_steps:
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            rows = validate(model, method, cfg, store, splits["val"], device)
            sc = _score(rows)
            per = {s: float(np.mean([r["dice"] for r in rows if r["scenario"] == s])) for s in SCENARIO_ORDER}
            rec = {"step": step, "type": "val", "score": sc, **{f"dice_{k}": v for k, v in per.items()},
                   "elapsed_min": (time.time() - t_start) / 60}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            log(f"[{method} s{seed}] step {step:>6d} val score={sc:.4f} "
                + " ".join(f"{k}={v:.4f}" for k, v in per.items())
                + f" ({(time.time()-t_start)/60:.1f} min)")
            if sc > best_score:
                best_score, best_step = sc, step
                torch.save({"model": model.state_dict(), "step": step, "score": sc,
                            "method": method, "seed": seed, "cfg": meta},
                           os.path.join(run_dir, "best.pt"))

        if step == cfg.max_steps:
            break

        # ---- 一个优化器更新 ----
        batch = make_batch(store, splits["train"], seed, step, cfg.patch,
                           cfg.tumor_center_prob, cfg.batch_size)
        b = _to_device(batch, device)
        if method == "A":
            loss, parts = forward_loss_A(model, b["mri"], b["avail"], b["y1"], cfg)
        else:
            loss, parts = forward_loss_B(model, b["mri"], b["avail"], b["y1"], seed, step, cfg, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        opt.step()

        if step % 100 == 0:
            rec = {"step": step, "type": "train", "loss": float(loss.detach()),
                   **{k: float(v) for k, v in parts.items()},
                   "elapsed_min": (time.time() - t_start) / 60}
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    torch.save({"model": model.state_dict(), "step": cfg.max_steps, "score": best_score,
                "method": method, "seed": seed, "cfg": meta},
               os.path.join(run_dir, "final.pt"))
    fh.close()
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    result = {"method": method, "seed": seed, "params": n_par, "best_score": best_score,
              "best_step": best_step, "train_minutes": (time.time() - t_start) / 60,
              "peak_gpu_gb": peak, "run_dir": run_dir}
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"[{method} s{seed}] 完成 best={best_score:.4f}@{best_step} "
        f"用时{(time.time()-t_start)/60:.1f}min 峰值显存{peak:.2f}GB")
    return result


def load_checkpoint(run_dir: str, which: str = "best.pt"):
    ck = torch.load(os.path.join(run_dir, which), map_location="cpu", weights_only=False)
    return ck
