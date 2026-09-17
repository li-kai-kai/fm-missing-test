#!/usr/bin/env python3
"""阶段三：冻结配置后测试 / 阶段四：多种子复验（方案 §6、§7）。

产物：
  predictions/<case>/<method>_seed<k>_<scenario>.nii.gz   保留空间信息
  metrics_per_case.csv   patient_id, model, seed, scenario, 指标, 耗时
  summary.csv
  bootstrap.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import Config, SCENARIO_ORDER, PRIMARY_SCENARIO, load_config
from fmexp.data import VolumeStore, avail_vector, decode_segmentation
from fmexp.infer import init_noise_for_case, predict_volume_A, predict_volume_B
from fmexp.metrics import segmentation_metrics, region_masks, gt_hd95_cache, paired_bootstrap, summarize
from fmexp.train import load_checkpoint
from fmexp.unet import build_model

METRIC_COLS = ["dice", "recall", "hd95", "pred_voxels", "gt_voxels",
               "pred_empty", "gt_empty", "complete_miss", "hd95_failed", "both_empty"]


def get_affine(cfg, cid):
    """与预处理一致的 RAS 仿射（用于把预测写回空间）。"""
    p = os.path.join(cfg.data_root, cid, f"{cid}_seg.nii.gz")
    return nib.as_closest_canonical(nib.load(p)).affine


def evaluate_split(method, seed, which_ckpt, cfg, splits, split_name, cases,
                   cache_dir, out_dir, device, save_predictions=True, tag="", run_tag=""):
    run_dir = os.path.join(out_dir, "runs", f"{method}_seed{seed}{run_tag}")
    run_cfg = load_config(os.path.join(run_dir, "config.yaml"))
    if run_cfg.task != cfg.task:
        raise ValueError("运行配置与预处理任务不一致")
    cfg = run_cfg
    ck = load_checkpoint(run_dir, which_ckpt)
    model = build_model(method, cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[{method} seed={seed}] 载入 {which_ckpt} (step={ck['step']}, val_score={ck['score']:.4f})")

    store = VolumeStore(cache_dir, cases, cfg.task)
    pred_root = os.path.join(out_dir, "predictions")
    if cfg.n_classes > 2 or split_name != "test":
        pred_root = os.path.join(pred_root, split_name)
    rows = []
    for cid in cases:
        case = store[cid]
        mri_full = torch.from_numpy(case.img).to(device)
        gt = case.seg
        affine = get_affine(cfg, cid) if save_predictions else None
        noise = init_noise_for_case(cid, gt.shape, cfg.eval_seed, device, cfg.n_classes) if method == "B" else None
        # 真实 mask 一侧的距离变换只算一次，供该病例三个场景共用
        gt_caches = {region: gt_hd95_cache(mask) for region, mask in region_masks(gt, cfg.n_classes).items()}

        for scen in cfg.scenarios:
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)     # 归一化之后置零（与训练一致）
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if method == "A":
                pred = predict_volume_A(model, mri, av, cfg)
                if cfg.n_classes == 2:
                    pred = (pred > 0.5).astype(np.uint8)
            else:
                pred = predict_volume_B(model, mri, av, cfg, noise)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

            for m in segmentation_metrics(pred, gt, cfg.n_classes, gt_caches=gt_caches):
                m.update({"patient_id": cid, "model": method, "seed": seed, "scenario": scen,
                          "seconds": dt, "peak_gpu_gb": peak, "split": split_name, "ckpt": which_ckpt})
                rows.append(m)
                print(f"  {cid} {scen} {m.get('region', 'WT')}: Dice={m['dice']:.4f} "
                      f"HD95={m['hd95']:.2f} 用时={dt:.1f}s 漏检={m['complete_miss']}")

            if save_predictions:
                d = os.path.join(pred_root, cid)
                os.makedirs(d, exist_ok=True)
                fn = os.path.join(d, f"{method}_seed{seed}{run_tag}{tag}_{scen}.nii.gz")
                nib.save(nib.Nifti1Image(decode_segmentation(pred, cfg.n_classes).astype(np.uint8), affine), fn)
    return rows


def write_csv(rows, path, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_summary(rows):
    """每个 (model, seed, scenario) 的汇总 + 配对差值。"""
    out = []
    keys = sorted({(r["model"], r["seed"], r["scenario"], r.get("region", "")) for r in rows})
    for model, seed, scen, region in keys:
        sub = [r for r in rows if r["model"] == model and r["seed"] == seed
               and r["scenario"] == scen and r.get("region", "") == region]
        s = {
            "model": model, "seed": seed, "scenario": scen, "n": len(sub),
            "dice_mean": summarize([r["dice"] for r in sub])["mean"],
            "dice_std": summarize([r["dice"] for r in sub])["std"],
            "recall_mean": summarize([r["recall"] for r in sub])["mean"],
            "hd95_mean": summarize([r["hd95"] for r in sub])["mean"],
            "hd95_failures": sum(1 for r in sub if r["hd95_failed"]),
            "complete_miss": sum(1 for r in sub if r["complete_miss"]),
            "both_empty": sum(1 for r in sub if r["both_empty"]),
            "seconds_mean": float(np.mean([r["seconds"] for r in sub])),
            "peak_gpu_gb_mean": float(np.mean([r["peak_gpu_gb"] for r in sub])),
        }
        if any(r.get("region") for r in rows):
            s["region"] = region
        out.append(s)
    return out


def paired_diffs(rows):
    """逐病例配对差值 B-A（方案 §7.2：先分别计算，再对病例取平均）。"""
    res = {}
    seeds = sorted({r["seed"] for r in rows})
    for seed in seeds:
        for scen, region in sorted({(r["scenario"], r.get("region", "")) for r in rows}):
            a = {r["patient_id"]: r["dice"] for r in rows
                 if r["model"] == "A" and r["seed"] == seed and r["scenario"] == scen
                 and r.get("region", "") == region}
            b = {r["patient_id"]: r["dice"] for r in rows
                 if r["model"] == "B" and r["seed"] == seed and r["scenario"] == scen
                 and r.get("region", "") == region}
            common = sorted(set(a) & set(b))
            if not common:
                continue
            diffs = np.array([b[c] - a[c] for c in common])
            res[f"{seed}|{scen}" + (f"|{region}" if region else "")] = {
                "n": len(common),
                "mean_diff_B_minus_A": float(diffs.mean()),
                "per_case": {c: float(b[c] - a[c]) for c in common},
                **{f"boot_{k}": v for k, v in paired_bootstrap(diffs, 2000, seed=42).items()},
            }
    return res


FLOAT_COLS = ("dice", "recall", "hd95", "seconds", "peak_gpu_gb",
              "pred_voxels", "gt_voxels")
BOOL_COLS = ("pred_empty", "gt_empty", "complete_miss", "hd95_failed", "both_empty")


def coerce_row(r):
    """CSV 读回来全是字符串，必须还原类型。

    尤其布尔列：字符串 "False" 是真值，不还原会把 complete_miss 全部算成 True。
    """
    r = dict(r)
    r["seed"] = int(r["seed"])
    for k in FLOAT_COLS:
        if k in r and r[k] != "":
            try:
                r[k] = float(r[k])
            except ValueError:
                r[k] = float("nan")
    for k in BOOL_COLS:
        if k in r:
            r[k] = str(r[k]).strip().lower() == "true"
    return r


def rebuild_combined(out):
    """把所有已存在的逐种子结果合并成 metrics_per_case.csv / summary.csv / bootstrap.json。

    这样多训练种子的汇总不重跑推理，只做合并。
    """
    import glob
    cols = ["patient_id", "model", "seed", "scenario", "region", "split", "ckpt"] + METRIC_COLS + \
           ["seconds", "peak_gpu_gb"]
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "metrics_per_case_seed*.csv"))):
        with open(p) as f:
            rows.extend(coerce_row(r) for r in csv.DictReader(f))
    if not rows:
        return [], [], {}
    unique = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in ("patient_id", "model", "seed", "scenario", "region", "split", "ckpt"))
        if key in unique and any(str(unique[key].get(k)) != str(r.get(k)) for k in METRIC_COLS):
            raise ValueError(f"重复评价结果不一致: {key}")
        unique[key] = r
    rows = list(unique.values())
    write_csv(rows, os.path.join(out, "metrics_per_case.csv"), cols)
    summary = build_summary(rows)
    write_csv(summary, os.path.join(out, "summary.csv"), list(summary[0].keys()))
    diffs = paired_diffs(rows)
    with open(os.path.join(out, "bootstrap.json"), "w") as f:
        json.dump(diffs, f, indent=2, ensure_ascii=False)
    return rows, summary, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--methods", default="A,B")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-save-pred", action="store_true")
    ap.add_argument("--run-tag", default="", help="运行目录后缀，用于调参实验")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    cfg = load_config(os.path.join(out, "config_preprocess.yaml"))
    splits = json.load(open(os.path.join(out, "splits.json")))
    cases = splits[args.split]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    all_rows = []
    # 新任务以及验证/调参结果按 split 和运行标签隔离。
    isolated = cfg.n_classes > 2 or (args.split != "test" and not args.run_tag)
    metrics_out = (os.path.join(out, "evaluation", args.split, args.run_tag or "main")
                   if isolated else out)
    cols = ["patient_id", "model", "seed", "scenario", "region", "split", "ckpt"] + METRIC_COLS + \
           ["seconds", "peak_gpu_gb"]
    t0 = time.time()
    for method in args.methods.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            rows = evaluate_split(method, seed, args.ckpt, cfg, splits, args.split, cases,
                                  os.path.join(out, "cache"), out, device,
                                  save_predictions=not args.no_save_pred, tag=args.tag,
                                  run_tag=args.run_tag)
            all_rows.extend(rows)
            if isolated:
                write_csv(rows, os.path.join(metrics_out, f"metrics_per_case_seed{seed}_{method}.csv"), cols)

    seeds_tag = "_".join(str(s) for s in sorted({int(s) for s in args.seeds.split(",")}))
    if isolated:
        _, summary, _ = rebuild_combined(metrics_out)
    elif args.run_tag:
        # 调参实验的指标单独存放，绝不能进入主结果的合并 glob（metrics_per_case_seed*.csv）
        fname = f"metrics_lr{args.run_tag.replace('/', '_')}.csv"
        write_csv(all_rows, os.path.join(out, fname), cols)
        summary = build_summary(all_rows)
    else:
        write_csv(all_rows, os.path.join(out, f"metrics_per_case_seed{seeds_tag}.csv"), cols)
        combined, summary, diffs = rebuild_combined(out)

    print("\n== 汇总 ==")
    print(f"{'model':>5} {'seed':>4} {'scenario':>8} {'Dice':>16} {'漏检':>5} {'HD95失败':>8} {'秒/例':>7}")
    for s in summary:
        print(f"{s['model']:>5} {s['seed']:>4} {s['scenario']:>8} {s.get('region', 'WT'):>3} "
              f"{s['dice_mean']:.4f}±{s['dice_std']:.4f} {s['complete_miss']:>5} "
              f"{s['hd95_failures']:>8} {s['seconds_mean']:>7.1f}")
    print(f"\n总用时 {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
