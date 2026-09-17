"""训练与验证（方案 §4 / §6 阶段二）。

checkpoint 选择规则（两模型一致）：固定验证安排下，各场景平均病例 Dice 的均值。
默认每例固定一个场景，按 split_seed 均衡分配，与模型/训练种子无关。
选定 best 后额外跑一次所有病例×所有场景，单独记录，不再选择 checkpoint。
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
from .metrics import segmentation_metrics
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
    y0, t = fm_noise(seed, step, mri.shape[0], mri.shape[2:], device, torch.float32,
                     n_ch=cfg.n_classes)
    yt = (1 - t) * y0 + t * y1
    target_v = y1 - y0
    tc = t.expand(t.shape[0], 1, *mri.shape[2:])
    x = torch.cat([mri, _expand_avail_t(avail, mri.shape), yt, tc], dim=1)
    pred_v = model(x)
    return loss_B(pred_v, target_v)


def validation_assignments(val_cases: Sequence[str], split_seed: int,
                           scenarios=SCENARIO_ORDER) -> Dict[str, str]:
    """固定、均衡的病例—场景安排；不依赖病例传入顺序或训练随机数。"""
    cases = sorted(val_cases)
    if len(set(cases)) != len(cases) or len(cases) < len(scenarios):
        raise ValueError("验证集必须无重复，且病例数不少于场景数")
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(cases))
    return {cases[int(i)]: scenarios[k % len(scenarios)]
            for k, i in enumerate(order)}


@torch.no_grad()
def validate(model, method: str, cfg: Config, store: VolumeStore,
             val_cases: Sequence[str], device,
             assignments: Dict[str, str] | None = None) -> List[dict]:
    """覆盖完整脑体积；未指定 assignments 时评估每例全部场景。"""
    was_training = model.training
    model.eval()
    rows = []
    for cid in val_cases:
        case = store[cid]
        mri_full = torch.from_numpy(case.img).to(device)   # [4,D,H,W]
        gt = case.seg
        noise = None
        if method == "B":
            noise = init_noise_for_case(cid, gt.shape, cfg.eval_seed, device, cfg.n_classes)
        for scen in ([assignments[cid]] if assignments is not None else cfg.scenarios):
            # 训练期选择 checkpoint 只按方案 §4.3 规定的平均病例 Dice，
            # 因此跳过 HD95（那一步的全脑距离变换是评价里最贵的一环）。
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)              # 归一化之后置零被隐藏模态
            t0 = time.time()
            if method == "A":
                prob = predict_volume_A(model, mri, av, cfg)
                pred = (prob > 0.5).astype(np.uint8) if cfg.n_classes == 2 else prob
            else:
                pred = predict_volume_B(model, mri, av, cfg, noise)
            dt = time.time() - t0
            for m in segmentation_metrics(pred, gt, cfg.n_classes, fast=True):
                m.update({"case_id": cid, "scenario": scen, "seconds": dt})
                rows.append(m)
    model.train(was_training)
    return rows


def _score(rows: Sequence[dict]) -> float:
    per = {}
    for r in rows:
        per.setdefault((r["scenario"], r.get("region", "WT")), []).append(r["dice"])
    return float(np.mean([np.mean(v) for v in per.values()]))


def _n_predictions(rows):
    return len({(r["case_id"], r["scenario"]) for r in rows})


def train_one(method: str, cfg: Config, seed: int, splits: dict,
              cache_dir: str, run_dir: str, device, log=print) -> dict:
    if cfg.val_every <= 0 or cfg.max_steps <= 0:
        raise ValueError("val_every 和 max_steps 必须为正数")
    if cfg.val_mode not in ("balanced", "full"):
        raise ValueError(f"未知 val_mode: {cfg.val_mode}")
    fixed_assignments = validation_assignments(splits["val"], cfg.split_seed, cfg.scenarios)
    assignments = fixed_assignments if cfg.val_mode == "balanced" else None
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    store = VolumeStore(cache_dir, splits["train"] + splits["val"], cfg.task)
    model = build_model(method, cfg).to(device)
    n_par = n_params(model)
    log(f"[{method} seed={seed}] 参数量 = {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path = os.path.join(run_dir, "log.jsonl")
    fh = open(log_path, "w")
    meta = {
        "method": method, "seed": seed, "params": n_par,
        "task": cfg.task, "n_classes": cfg.n_classes, "scenarios": cfg.scenarios,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
        "patch": cfg.patch, "max_steps": cfg.max_steps, "val_every": cfg.val_every,
        "fm_steps": cfg.fm_steps, "sw_window": cfg.sw_window, "sw_overlap": cfg.sw_overlap,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "train_cases": splits["train"], "val_cases": splits["val"],
        "val_mode": cfg.val_mode,
        "val_assignments": assignments,
        "checkpoint_selection": "mean_of_scenario_region_mean_dice",
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    best_score, best_step = -np.inf, -1
    t_start = time.time()
    validation_seconds = 0.0
    model.train()

    # 每个场景一例，仅检查完整推理链路；不参与 best checkpoint 选择。
    smoke_cases = [next(c for c, s in fixed_assignments.items() if s == scen)
                   for scen in cfg.scenarios]
    t_val = time.time()
    smoke_rows = validate(model, method, cfg, store, smoke_cases, device, fixed_assignments)
    validation_seconds += time.time() - t_val
    fh.write(json.dumps({"step": 0, "type": "smoke", "score": _score(smoke_rows),
                         "n_predictions": _n_predictions(smoke_rows), "rows": smoke_rows,
                         "elapsed_min": (time.time() - t_start) / 60}) + "\n")
    fh.flush()
    log(f"[{method} s{seed}] step 0 smoke: {_n_predictions(smoke_rows)} 个病例—场景，不参与 checkpoint 选择")

    for step in range(cfg.max_steps + 1):
        if step > 0 and (step % cfg.val_every == 0 or step == cfg.max_steps):
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            t_val = time.time()
            rows = validate(model, method, cfg, store, splits["val"], device, assignments)
            validation_seconds += time.time() - t_val
            sc = _score(rows)
            per = {s: float(np.mean([r["dice"] for r in rows if r["scenario"] == s])) for s in cfg.scenarios}
            rec = {"step": step, "type": "val", "score": sc, **{f"dice_{k}": v for k, v in per.items()},
                   "val_mode": cfg.val_mode, "n_predictions": _n_predictions(rows),
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
                           cfg.tumor_center_prob, cfg.batch_size,
                           n_classes=cfg.n_classes, scenario_order=cfg.scenarios)
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
    # 全场景复核已经选定的 best，不能用这一次结果重新选择或改写其 selection score。
    ck = load_checkpoint(run_dir)
    model.load_state_dict(ck["model"])
    t_val = time.time()
    full_rows = validate(model, method, cfg, store, splits["val"], device)
    validation_seconds += time.time() - t_val
    full_score = _score(full_rows)
    with open(os.path.join(run_dir, "best_full_validation.json"), "w") as f:
        json.dump({"split": "val", "step": best_step, "selection_score": best_score,
                   "score": full_score, "rows": full_rows}, f, indent=2)
    fh.write(json.dumps({"step": best_step, "type": "val_full", "score": full_score,
                         "n_predictions": _n_predictions(full_rows),
                         "elapsed_min": (time.time() - t_start) / 60}) + "\n")
    fh.close()
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    result = {"method": method, "seed": seed, "params": n_par, "best_score": best_score,
              "best_step": best_step, "train_minutes": (time.time() - t_start) / 60,
              "validation_minutes": validation_seconds / 60,
              "best_full_val_score": full_score, "val_mode": cfg.val_mode,
              "peak_gpu_gb": peak, "run_dir": run_dir}
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"[{method} s{seed}] 完成 best={best_score:.4f}@{best_step} "
        f"全场景验证={full_score:.4f} 总用时{(time.time()-t_start)/60:.1f}min "
        f"其中验证{validation_seconds/60:.1f}min 峰值显存{peak:.2f}GB")
    return result


def load_checkpoint(run_dir: str, which: str = "best.pt"):
    ck = torch.load(os.path.join(run_dir, which), map_location="cpu", weights_only=False)
    return ck
