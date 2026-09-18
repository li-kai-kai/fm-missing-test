#!/usr/bin/env python3
"""失败归因诊断入口（只读既有 checkpoint，不改动冻结配置，不覆盖主实验结果）。

用法示例:
    python3 run_diagnostics.py --out experiment_multiclass --check location
    python3 run_diagnostics.py --out experiment_multiclass --check fit --method B
    python3 run_diagnostics.py --out experiment_multiclass --check time --method B
    python3 run_diagnostics.py --out experiment_multiclass --check condition --method B
    python3 run_diagnostics.py --out experiment_multiclass --check steps --method B
    python3 run_diagnostics.py --out experiment_multiclass --check all
    python3 run_diagnostics.py --out experiment_multiclass --report

产物全部落在 <out>/diagnostics/ 下，与 runs/、evaluation/、predictions/ 完全隔离。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import load_config
from fmexp.data import VolumeStore, avail_vector, load_case
from fmexp.diagnostics import (conditioning_variants, connected_component_stats,
                               distance_to_support, endpoint_margin, error_location,
                               final_state, mask_to_support,
                               trajectory_diagnostic, velocity_mse_by_t,
                               steps_sweep, _region_dice)
from fmexp.infer import init_noise_for_case, predict_volume_A, predict_volume_B
from fmexp.train import load_checkpoint
from fmexp.unet import build_model

REGIONS = ("WT", "TC", "ET")


# ---------------------------------------------------------------- 公共

def diag_dir(out: str) -> str:
    d = os.path.join(out, "diagnostics")
    os.makedirs(d, exist_ok=True)
    return d


def dump(out: str, name: str, payload: dict) -> str:
    p = os.path.join(diag_dir(out), f"{name}.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  -> {p}")
    return p


def load_run(out: str, method: str, seed: int, ckpt: str, run_dir: str | None, device):
    run_dir = run_dir or os.path.join(out, "runs", f"{method}_seed{seed}")
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    ck = load_checkpoint(run_dir, ckpt)
    model = build_model(method, cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[{method} seed={seed}] 载入 {run_dir}/{ckpt} step={ck['step']}")
    return model, cfg


def read_pred_nifti(path: str) -> np.ndarray:
    """预测 NIfTI 存的是原始标签 0/1/2/4，转回内部类别 0/1/2/3。"""
    a = np.asanyarray(nib.load(path).dataobj)
    return np.where(a == 4, 3, a).astype(np.uint8)


def full_volume_dice(model, method, cfg, mri_full, av, gt, noise, scen, device,
                     with_mask=True, support=None, chunk=32):
    mri = mri_full * av.view(4, 1, 1, 1)
    if method == "A":
        prob = predict_volume_A(model, mri, av, cfg, chunk=chunk)
        pred = (prob > 0.5).astype(np.uint8) if cfg.n_classes == 2 else prob
    else:
        pred = predict_volume_B(model, mri, av, cfg, noise, chunk=chunk)
    out = {"dice": {r: _region_dice(pred, gt, r) for r in REGIONS},
           "components": connected_component_stats(pred > 0)}
    if with_mask and support is not None:
        pm = mask_to_support(pred, support)
        out["dice_masked"] = {r: _region_dice(pm, gt, r) for r in REGIONS}
        out["components_masked"] = connected_component_stats(pm > 0)
    return out


# ---------------------------------------------------------------- 检查 3a：错误位置

def check_location(out: str, methods, seeds, scenarios, run_tag="", tag="", with_distance=True):
    """纯 CPU：从已保存的预测 NIfTI 统计脑内/脑外错误、混淆、碎块。"""
    splits = json.load(open(os.path.join(out, "splits.json")))
    cache = os.path.join(out, "cache")
    pred_root = os.path.join(out, "predictions", "test")
    cases = splits["test"]
    store = VolumeStore(cache, cases, "multiclass_missing_one")
    result = {"cases": cases, "per_method": {}}
    # 到脑支撑的距离只取决于病例，与模型/种子/场景无关，全实验只算一次
    dists = {}
    if with_distance:
        t0 = time.time()
        dists = {cid: distance_to_support(store[cid].support) for cid in cases}
        print(f"  距离变换完成 ({time.time()-t0:.0f}s)")
    for method in methods:
        for seed in seeds:
            t0 = time.time()
            agg = {}
            for scen in scenarios:
                tot = {}
                conf = np.zeros((4, 4), dtype=np.int64)
                n_comp, out_dist = [], []
                for cid in cases:
                    fp = os.path.join(pred_root, cid,
                                      f"{method}_seed{seed}{run_tag}{tag}_{scen}.nii.gz")
                    if not os.path.exists(fp):
                        continue
                    pred = read_pred_nifti(fp)
                    case = store[cid]
                    r = error_location(pred, case.seg, case.support, 4,
                                       dists.get(cid))
                    for k in ("tp", "fn", "fp_in", "fp_out", "gt_voxels"):
                        tot[k] = tot.get(k, 0) + r[k]
                    conf += np.array(r["confusion"], dtype=np.int64)
                    n_comp.append(r["components"]["n_components"])
                    if "fp_out_distance" in r:
                        out_dist.append(r["fp_out_distance"])
                if not tot:
                    continue
                fp = tot["fp_in"] + tot["fp_out"]
                agg[scen] = {
                    **tot,
                    "recall": tot["tp"] / max(tot["tp"] + tot["fn"], 1),
                    "precision": tot["tp"] / max(tot["tp"] + fp, 1),
                    "fp_out_frac": tot["fp_out"] / max(fp, 1),
                    "confusion": conf.tolist(),
                    "components_median": float(np.median(n_comp)) if n_comp else 0.0,
                    "components_max": int(np.max(n_comp)) if n_comp else 0,
                    "fp_out_distance_median": float(np.median(
                        [d["median"] for d in out_dist])) if out_dist else None,
                    "fp_out_within_10mm": float(np.mean(
                        [d["within_10mm"] for d in out_dist])) if out_dist else None,
                }
            result["per_method"][f"{method}_seed{seed}"] = {
                "scenarios": agg, "seconds": time.time() - t0}
            print(f"  位置检查 {method} seed{seed} 完成 ({time.time()-t0:.0f}s)")
    return result


# ---------------------------------------------------------------- 检查 1a：训练病例拟合

def check_fit(out, method, seed, ckpt, run_dir, device, n_train=8, n_test=5,
              scenarios=None, chunk=32):
    """在训练病例与测试病例上跑同样的完整生成，直接比较泛化差距。"""
    splits = json.load(open(os.path.join(out, "splits.json")))
    model, cfg = load_run(out, method, seed, ckpt, run_dir, device)
    scenarios = scenarios or cfg.scenarios
    cache = os.path.join(out, "cache")
    tr = splits["train"][:n_train]
    te = splits["test"][:n_test]
    store = VolumeStore(cache, tr + te, cfg.task)
    res = {"method": method, "seed": seed, "ckpt": ckpt, "run_dir": run_dir,
           "n_train": len(tr), "n_test": len(te), "groups": {}}
    for gname, gcases in (("train", tr), ("test", te)):
        rows = []
        for cid in gcases:
            case = store[cid]
            mri_full = torch.from_numpy(case.img).to(device)
            noise = (init_noise_for_case(cid, case.seg.shape, cfg.eval_seed, device,
                                         cfg.n_classes) if method == "B" else None)
            for scen in scenarios:
                av = torch.from_numpy(avail_vector(scen)).to(device)
                r = full_volume_dice(model, method, cfg, mri_full, av, case.seg,
                                     noise, scen, device, support=case.support, chunk=chunk)
                r.update({"case_id": cid, "scenario": scen})
                rows.append(r)
        agg = {}
        for region in REGIONS:
            raw = [r["dice"][region] for r in rows]
            msk = [r["dice_masked"][region] for r in rows]
            agg[region] = {
                "dice_mean": float(np.mean(raw)), "dice_masked_mean": float(np.mean(msk)),
                "per_case": raw,
            }
        agg["components_median"] = float(np.median(
            [r["components"]["n_components"] for r in rows]))
        agg["components_masked_median"] = float(np.median(
            [r["components_masked"]["n_components"] for r in rows]))
        res["groups"][gname] = agg
        print(f"  {gname}: WT Dice={agg['WT']['dice_mean']:.4f} "
              f"(掩蔽后 {agg['WT']['dice_masked_mean']:.4f}) "
              f"碎块中位={agg['components_median']:.0f}")
    return res


# ---------------------------------------------------------------- 检查 2a/2b/2d

def check_time(out, method, seed, ckpt, run_dir, device, n_batches=16, n_cases=3,
               chunk=32):
    splits = json.load(open(os.path.join(out, "splits.json")))
    model, cfg = load_run(out, method, seed, ckpt, run_dir, device)
    cache = os.path.join(out, "cache")
    res = {"method": method, "seed": seed}

    # 2a：patch 上沿理想路径的速度 MSE（分 t、分类别）。
    # 用验证病例（未参与训练）取样，衡量的是学到的场在留出数据上的行为。
    pool = splits["val"]
    store = VolumeStore(cache, pool, cfg.task)
    t0 = time.time()
    res["velocity_mse_by_t"] = velocity_mse_by_t(model, store, pool, cfg,
                                                 device, n_batches=n_batches, chunk=chunk)
    res["pool"] = "val"
    print(f"  2a 分时间速度 MSE 完成 ({time.time()-t0:.0f}s)")
    del store
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # 2b/2d：实际轨迹 vs 理想路径
    cases = splits["test"][:n_cases]
    store = VolumeStore(cache, cases, cfg.task)
    traj = []
    for cid in cases:
        case = store[cid]
        mri_full = torch.from_numpy(case.img).to(device)
        noise = init_noise_for_case(cid, case.seg.shape, cfg.eval_seed, device, cfg.n_classes)
        for scen in cfg.scenarios:
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)
            t1 = time.time()
            d = trajectory_diagnostic(model, mri, av, cfg, noise, case.seg, chunk=chunk)
            d.update({"case_id": cid, "scenario": scen})
            traj.append(d)
            print(f"  2b {cid} {scen}: 末步 Dice={d['final_dice']['WT']:.4f} "
                  f"({time.time()-t1:.0f}s)")
    res["trajectory"] = traj
    # 按 t 聚合成一张表
    agg = {}
    for d in traj:
        for row in d["rows"]:
            a = agg.setdefault(row["step"], {"t": row["t"], "n": 0, "drift": 0.0,
                                             "v_traj": 0.0, "v_ideal": 0.0,
                                             "dice_traj": 0.0, "dice_ideal": 0.0})
            a["n"] += 1
            a["drift"] += row["drift_mse"]
            a["v_traj"] += row["v_mse_ontraj"]
            a["v_ideal"] += row["v_mse_onideal"]
            a["dice_traj"] += row["dice_traj"]
            a["dice_ideal"] += row["dice_ideal"]
    for a in agg.values():
        for k in ("drift", "v_traj", "v_ideal", "dice_traj", "dice_ideal"):
            a[k] /= a["n"]
    res["trajectory_agg"] = [agg[k] for k in sorted(agg)]
    res["var_v"] = traj[0]["var_v"] if traj else None
    return res


# ---------------------------------------------------------------- 检查 2c：条件消融

def check_condition(out, method, seed, ckpt, run_dir, device, n_cases=6,
                    scenarios=("missing_t1ce",), n_noise=3, chunk=32):
    splits = json.load(open(os.path.join(out, "splits.json")))
    model, cfg = load_run(out, method, seed, ckpt, run_dir, device)
    cache = os.path.join(out, "cache")
    cases = splits["test"][:n_cases]
    others = splits["test"][n_cases:n_cases + 1] or splits["test"][:1]
    store = VolumeStore(cache, sorted(set(cases) | set(others)), cfg.task)
    rng = np.random.default_rng(0)
    res = {"method": method, "seed": seed, "cases": cases, "per_scenario": {}}
    for scen in scenarios:
        agg = {k: {r: [] for r in REGIONS} for k in
               ("real", "zero", "swap_case", "shuffle_vox")}
        noise_sweep, mri_sweep = [], []
        for cid in cases:
            case = store[cid]
            mri_full = case.img
            other = store[others[0]].img
            noise = init_noise_for_case(cid, case.seg.shape, cfg.eval_seed, device, cfg.n_classes)
            av_np = avail_vector(scen)
            variants = conditioning_variants(mri_full, case.support, cid, other,
                                             cfg.eval_seed, rng)
            for name, vol in variants.items():
                av = torch.from_numpy(av_np).to(device)
                m = torch.from_numpy(np.ascontiguousarray(vol)).to(device)
                r = full_volume_dice(model, method, cfg, m, av, case.seg, noise, scen,
                                     device, with_mask=False, chunk=chunk)
                for region in REGIONS:
                    agg[name][region].append(r["dice"][region])
            if scen == scenarios[0]:
                # 双向分解：固定 MRI 变噪声 vs 固定噪声变 MRI
                base_av = torch.from_numpy(av_np).to(device)
                m_fix = torch.from_numpy(np.ascontiguousarray(mri_full)).to(device)
                for k in range(n_noise):
                    nz = init_noise_for_case(cid, case.seg.shape, 1000 + k, device, cfg.n_classes)
                    d = full_volume_dice(model, method, cfg, m_fix, base_av, case.seg,
                                         nz, scen, device, with_mask=False, chunk=chunk)
                    noise_sweep.append(d["dice"]["WT"])
                for oc in others:
                    m_var = torch.from_numpy(np.ascontiguousarray(store[oc].img)).to(device)
                    d = full_volume_dice(model, method, cfg, m_var, base_av, case.seg,
                                         noise, scen, device, with_mask=False, chunk=chunk)
                    mri_sweep.append(d["dice"]["WT"])
        res["per_scenario"][scen] = {
            k: {r: float(np.mean(v[r])) for r in REGIONS} for k, v in agg.items()}
        if noise_sweep:
            res["noise_sweep_WT"] = noise_sweep
            res["mri_sweep_WT"] = mri_sweep
        print(f"  条件消融 {scen}: " + " ".join(
            f"{k}={res['per_scenario'][scen][k]['WT']:.4f}"
            for k in ("real", "zero", "swap_case", "shuffle_vox")))
    return res


# ---------------------------------------------------------------- 检查 4a：步数

def check_steps(out, method, seed, ckpt, run_dir, device, n_cases=5,
                scenarios=("missing_t1ce",), steps_list=(8, 16, 32, 64, 128),
                chunk=32):
    splits = json.load(open(os.path.join(out, "splits.json")))
    model, cfg = load_run(out, method, seed, ckpt, run_dir, device)
    cache = os.path.join(out, "cache")
    cases = splits["test"][:n_cases]
    store = VolumeStore(cache, cases, cfg.task)
    res = {"method": method, "seed": seed, "cases": cases, "per_scenario": {}}
    for scen in scenarios:
        per_case, agg = [], {}
        for cid in cases:
            case = store[cid]
            mri_full = torch.from_numpy(case.img).to(device)
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)
            noise = init_noise_for_case(cid, case.seg.shape, cfg.eval_seed, device, cfg.n_classes)
            r = steps_sweep(model, mri, av, cfg, noise, case.seg, steps_list, chunk=chunk)
            per_case.append({"case_id": cid, **{str(k): v for k, v in r.items()}})
            for s, v in r.items():
                a = agg.setdefault(s, {r_: [] for r_ in REGIONS})
                a.setdefault("n_comp", [])
                for region in REGIONS:
                    a[region].append(v["dice"][region])
                a["n_comp"].append(v["components"]["n_components"])
        summary = {str(s): {r: float(np.mean(v[r])) for r in REGIONS} for s, v in agg.items()}
        for s, v in agg.items():
            summary[str(s)]["n_components_median"] = float(np.median(v["n_comp"]))
        res["per_scenario"][scen] = {"summary": summary, "per_case": per_case}
        print(f"  步数扫描 {scen}: " + " ".join(
            f"{s}步={summary[str(s)]['WT']:.4f}/{summary[str(s)]['n_components_median']:.0f}块"
            for s in steps_list))
    return res


# ---------------------------------------------------------------- 诊断重训的测试集评估

def check_evaltest(out, method, seed, ckpt, run_dir, device, n_cases=None, chunk=32):
    """在完整测试划分上评估某个 run（含 diagnostics/runs 下的诊断重训）。

    走的是与 run_eval.py 相同的整卷推理与 Dice 口径，但产物只写 diagnostics/，
    绝不碰主实验的 evaluation/。
    """
    splits = json.load(open(os.path.join(out, "splits.json")))
    model, cfg = load_run(out, method, seed, ckpt, run_dir, device)
    cases = splits["test"][:n_cases] if n_cases else splits["test"]
    store = VolumeStore(os.path.join(out, "cache"), cases, cfg.task)
    rows = []
    for cid in cases:
        case = store[cid]
        mri_full = torch.from_numpy(case.img).to(device)
        noise = (init_noise_for_case(cid, case.seg.shape, cfg.eval_seed, device,
                                     cfg.n_classes) if method == "B" else None)
        for scen in cfg.scenarios:
            av = torch.from_numpy(avail_vector(scen)).to(device)
            r = full_volume_dice(model, method, cfg, mri_full, av, case.seg, noise,
                                 scen, device, with_mask=False, chunk=chunk)
            r.update({"case_id": cid, "scenario": scen})
            rows.append(r)
    summary = {}
    for scen in cfg.scenarios:
        sub = [r for r in rows if r["scenario"] == scen]
        summary[scen] = {reg: float(np.mean([r["dice"][reg] for r in sub])) for reg in REGIONS}
        summary[scen]["components_median"] = float(np.median(
            [r["components"]["n_components"] for r in sub]))
    overall = {reg: float(np.mean([r["dice"][reg] for r in rows])) for reg in REGIONS}
    print(f"  测试集 {len(cases)} 例 × {len(cfg.scenarios)} 场景: " +
          " ".join(f"{k}={v:.4f}" for k, v in overall.items()))
    label = run_dir or f"runs/{method}_seed{seed}"
    return {"method": method, "seed": seed, "run_dir": label, "cases": cases,
            "overall": overall, "per_scenario": summary,
            "per_case": [{"case_id": r["case_id"], "scenario": r["scenario"],
                          "dice": r["dice"]} for r in rows]}


# ---------------------------------------------------------------- 检查 2e：端点置信间隔

def check_endpoint(out, method, seed, ckpt, run_dir, device, n_cases=3,
                   scenarios=("missing_t1ce", "missing_flair"),
                   steps_list=(8, 16, 32, 64, 128), chunk=32):
    """终点连续状态的 max−second 间隔：判断 argmax 是不是在糊掉的区域里随机翻转。"""
    splits = json.load(open(os.path.join(out, "splits.json")))
    model, cfg = load_run(out, method, seed, ckpt, run_dir, device)
    store = VolumeStore(os.path.join(out, "cache"), splits["test"][:n_cases], cfg.task)
    res = {"method": method, "seed": seed, "cases": [], "per_case": []}
    for cid in splits["test"][:n_cases]:
        case = store[cid]
        res["cases"].append(cid)
        mri_full = torch.from_numpy(case.img).to(device)
        noise = init_noise_for_case(cid, case.seg.shape, cfg.eval_seed, device, cfg.n_classes)
        for scen in scenarios:
            av = torch.from_numpy(avail_vector(scen)).to(device)
            mri = mri_full * av.view(4, 1, 1, 1)
            row = {"case_id": cid, "scenario": scen, "by_steps": {}}
            for s in steps_list:
                y = final_state(model, mri, av, cfg, noise, steps=s, chunk=chunk)
                m = endpoint_margin(y, case.seg, case.support)
                m["dice_WT"] = _region_dice(
                    y.argmax(dim=0).cpu().numpy().astype(np.uint8), case.seg, "WT")
                row["by_steps"][str(s)] = m
            res["per_case"].append(row)
            print(f"  端点 {cid[-7:]} {scen}: " + " ".join(
                f"{s}步 Dice={row['by_steps'][str(s)]['dice_WT']:.3f} "
                f"中位间隔={row['by_steps'][str(s)]['margin_median']:.3f}"
                for s in steps_list))
    # 按步数聚合
    agg = {}
    for row in res["per_case"]:
        for s, m in row["by_steps"].items():
            a = agg.setdefault(s, {k: [] for k in m})
            for k, v in m.items():
                if v is not None:
                    a[k].append(v)
    res["summary"] = {s: {k: float(np.mean(v)) for k, v in a.items() if v}
                      for s, a in agg.items()}
    return res


# ---------------------------------------------------------------- main

CHECKS = ("fit", "time", "condition", "location", "steps", "endpoint", "evaltest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment_multiclass")
    ap.add_argument("--check", default="all", choices=CHECKS + ("all",))
    ap.add_argument("--method", default="B", choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--run-dir", default=None, help="显式指定 run 目录（诊断重训用）")
    ap.add_argument("--run-tag", default="", help="预测文件名里的运行后缀（location 用）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-cases", type=int, default=None)
    ap.add_argument("--steps-list", default="8,16,32,64,128")
    ap.add_argument("--report", action="store_true", help="只根据已有 JSON 渲染 diagnostics/report.md")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    if args.report:
        from fmexp.diag_report import write_diag_report
        p = write_diag_report(out)
        print(f"报告 -> {p}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== 诊断 {args.check} 设备={device} ==")
    checks = CHECKS if args.check == "all" else (args.check,)
    # 产物名：location 覆盖全部方法/种子，固定一个名字；重训用 run 目录名区分
    if args.run_dir:
        base = f"{args.method}_seed{args.seed}_" + os.path.basename(args.run_dir.rstrip("/"))
    else:
        base = f"{args.method}_seed{args.seed}{args.tag}"
    for c in checks:
        t0 = time.time()
        print(f"-- 检查 {c} --")
        if c == "location":
            payload = check_location(out, ["A", "B"], [0, 1, 2],
                                     ["missing_t1", "missing_t2", "missing_t1ce", "missing_flair"],
                                     run_tag=args.run_tag, tag=args.tag)
        elif c == "fit":
            payload = check_fit(out, args.method, args.seed, args.ckpt, args.run_dir, device,
                                n_train=args.n_cases or 8,
                                n_test=min(5, args.n_cases or 5))
        elif c == "time":
            payload = check_time(out, args.method, args.seed, args.ckpt, args.run_dir, device,
                                 n_cases=args.n_cases or 3)
        elif c == "condition":
            payload = check_condition(out, args.method, args.seed, args.ckpt, args.run_dir,
                                      device, n_cases=args.n_cases or 6)
        elif c == "steps":
            payload = check_steps(out, args.method, args.seed, args.ckpt, args.run_dir, device,
                                  n_cases=args.n_cases or 5,
                                  steps_list=[int(s) for s in args.steps_list.split(",")])
        elif c == "evaltest":
            payload = check_evaltest(out, args.method, args.seed, args.ckpt, args.run_dir,
                                     device, n_cases=args.n_cases)
        elif c == "endpoint":
            payload = check_endpoint(out, args.method, args.seed, args.ckpt, args.run_dir,
                                     device, n_cases=args.n_cases or 3,
                                     steps_list=[int(s) for s in args.steps_list.split(",")])
        payload["_check"] = c
        payload["_seconds"] = time.time() - t0
        dump(out, "location" if c == "location" else f"{c}_{base}", payload)
        print(f"-- 检查 {c} 用时 {(time.time()-t0)/60:.1f} min --")


if __name__ == "__main__":
    main()
