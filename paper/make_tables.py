#!/usr/bin/env python3
"""从实验产物直接生成论文用的 LaTeX 表格，避免手抄数字。

输入（只读，不修改任何实验产物）：
  experiment_multiclass/evaluation/test/main/summary.csv   四分类主结果
  experiment/summary.csv                                    二分类结果
  experiment_multiclass/diagnostics/*.json                  诊断读数

输出：
  paper/tables/*.tex
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "tables")

REGION_ORDER = ["WT", "TC", "ET"]
SCEN_ORDER = ["missing_t1", "missing_t2", "missing_t1ce", "missing_flair"]
SCEN_LABEL = {
    "missing_t1": "w/o T1",
    "missing_t2": "w/o T2",
    "missing_t1ce": "w/o T1ce",
    "missing_flair": "w/o FLAIR",
    "C0": "C0 (all)",
    "C1": "C1 (w/o T1ce)",
    "C2": "C2 (T2+FLAIR)",
}


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x, nd=3):
    return f"{x:.{nd}f}"


def multiclass_tables():
    """四分类：按 (model, scenario, region) 对三个训练种子聚合。"""
    rows = read_csv(os.path.join(ROOT, "experiment_multiclass/evaluation/test/main/summary.csv"))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["model"], r["scenario"], r["region"])
        for col in ("dice_mean", "hd95_mean", "seconds_mean", "recall_mean"):
            agg[k][col].append(float(r[col]))
        agg[k]["n"].append(int(r["n"]))

    lines = []
    add = lines.append
    add(r"\begin{table*}[t]")
    add(r"\centering")
    add(r"\caption{Four-class missing-modality segmentation on the BraTS test split "
        r"(20 patients, mean $\pm$ std over three training seeds). "
        r"Dice and HD95 are computed per patient then averaged; HD95 is in mm "
        r"and averages only over cases where both masks are non-empty.}")
    add(r"\label{tab:main}")
    add(r"\begin{tabular}{llcccccc}")
    add(r"\toprule")
    add(r"Missing & Region & \multicolumn{2}{c}{Dice $\uparrow$} & \multicolumn{2}{c}{HD95 $\downarrow$} "
        r"& \multicolumn{2}{c}{Sec./vol. $\downarrow$} \\")
    add(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    add(r"& & A (disc.) & B (FM) & A & B & A & B \\")
    add(r"\midrule")
    for scen in SCEN_ORDER:
        for region in REGION_ORDER:
            ka = ("A", scen, region)
            kb = ("B", scen, region)
            if ka not in agg:
                continue
            da, ds = mean_std(agg[ka]["dice_mean"])
            dba, dbs = mean_std(agg[kb]["dice_mean"])
            ha, _ = mean_std(agg[ka]["hd95_mean"])
            hb, _ = mean_std(agg[kb]["hd95_mean"])
            sa, _ = mean_std(agg[ka]["seconds_mean"])
            sb, _ = mean_std(agg[kb]["seconds_mean"])
            add(f"{SCEN_LABEL[scen]} & {region} & "
                f"${fnum(da)}\\pm{fnum(ds)}$ & ${fnum(dba)}\\pm{fnum(dbs)}$ & "
                f"{fnum(ha,1)} & {fnum(hb,1)} & "
                f"{fnum(sa,2)} & {fnum(sb,2)} \\\\")
        add(r"\addlinespace")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table*}")

    # 按区域合并缺失场景的紧凑版本
    agg2 = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["model"], r["region"])
        agg2[k]["dice_mean"].append(float(r["dice_mean"]))
        agg2[k]["seconds_mean"].append(float(r["seconds_mean"]))
        agg2[k]["hd95_mean"].append(float(r["hd95_mean"]))
    compact = [r"\begin{table}[t]", r"\centering",
               r"\caption{Region-wise summary over the four missing-modality "
               r"settings (mean over settings and seeds).}", r"\label{tab:region}",
               r"\begin{tabular}{lcccc}", r"\toprule",
               r"Region & A Dice & B Dice & A sec. & B sec. \\", r"\midrule"]
    for region in REGION_ORDER:
        ka, kb = ("A", region), ("B", region)
        da, ds = mean_std(agg2[ka]["dice_mean"])
        dba, dbs = mean_std(agg2[kb]["dice_mean"])
        sa, _ = mean_std(agg2[ka]["seconds_mean"])
        sb, _ = mean_std(agg2[kb]["seconds_mean"])
        compact.append(f"{region} & ${fnum(da)}\\pm{fnum(ds)}$ & ${fnum(dba)}\\pm{fnum(dbs)}$ "
                       f"& {fnum(sa,2)} & {fnum(sb,2)} \\\\")
    compact += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), "\n".join(compact)


def binary_table():
    rows = read_csv(os.path.join(ROOT, "experiment/summary.csv"))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["model"], r["scenario"])
        for col in ("dice_mean", "seconds_mean", "recall_mean"):
            agg[k][col].append(float(r[col]))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Binary whole-tumour task (three seeds, 20 test patients). "
             r"Flow matching stays within $\sim$0.10 Dice here, unlike the "
             r"four-class task in Table~\ref{tab:main}.}", r"\label{tab:binary}",
             r"\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Scenario & $A$ Dice & $B$ Dice & $A$ sec. & $B$ sec. \\", r"\midrule"]
    for scen in ["C0", "C1", "C2"]:
        da, ds = mean_std(agg[("A", scen)]["dice_mean"])
        dba, dbs = mean_std(agg[("B", scen)]["dice_mean"])
        sa, _ = mean_std(agg[("A", scen)]["seconds_mean"])
        sb, _ = mean_std(agg[("B", scen)]["seconds_mean"])
        lines.append(f"{SCEN_LABEL[scen]} & ${fnum(da)}\\pm{fnum(ds)}$ & "
                     f"${fnum(dba)}\\pm{fnum(dbs)}$ & {fnum(sa,2)} & {fnum(sb,2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


ARMS = [
    ("_fgw9", "B + foreground"),
    ("_tg1", "B + late-time"),
    ("_fgw9tg1", "B + both"),
]


def remedy_table():
    """修复臂的测试集结果，逐种子先算均值再对种子取 mean±std。"""
    ev = os.path.join(ROOT, "experiment_multiclass/evaluation/test")

    def per_seed(path, model=None):
        rows = read_csv(path)
        if model:
            rows = [r for r in rows if r["model"] == model]
        agg = defaultdict(list)
        for r in rows:
            agg[(r["seed"], r["region"])].append(float(r["dice"]))
        return {k: sum(v) / len(v) for k, v in agg.items()}

    main_p = os.path.join(ev, "main/metrics_per_case.csv")
    series = [("A (disc.)", per_seed(main_p, "A"), None),
              ("B (baseline)", per_seed(main_p, "B"), None)]
    notes = {}
    for tag, label in ARMS:
        p = os.path.join(ev, tag, "metrics_per_case.csv")
        if not os.path.exists(p):
            continue
        d = per_seed(p)
        seeds = sorted({k[0] for k in d})
        series.append((label, d, None))
        if len(seeds) < 3:
            notes[label] = len(seeds)
            print(f"注意：{label} 只评估了 {len(seeds)} 个种子")

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Effect of the two loss reweightings on the four-class "
             r"test split, averaged over the four missing-modality patterns. "
             r"Mean $\pm$ std across three training seeds of the per-seed "
             r"case-mean Dice. Neither reweighting recovers whole-tumour Dice; "
             r"foreground weighting trades it for a consistent gain on the "
             r"smallest class.}", r"\label{tab:remedy}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Configuration & WT & TC & ET \\", r"\midrule"]
    for label, d, _ in series:
        cells = []
        for region in REGION_ORDER:
            vals = [v for (s, r), v in d.items() if r == region]
            m, sd = mean_std(sorted(vals))
            cells.append(f"${fnum(m)}\\pm{fnum(sd)}$" if sd else f"${fnum(m)}$")
        star = r"$^{\dagger}$" if label in notes else ""
        lines.append(f"{label}{star} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\\[2pt] \footnotesize $^{\dagger}$ evaluated on a single "
                     r"training seed.")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def diag_numbers():
    """打印诊断读数，供正文引用（不直接生成表格）。"""
    d = os.path.join(ROOT, "experiment_multiclass/diagnostics")
    out = {}

    steps = json.load(open(os.path.join(d, "steps_B_seed0.json")))
    out["steps_sweep"] = steps.get("per_scenario")

    ep = json.load(open(os.path.join(d, "endpoint_B_seed0.json")))
    out["endpoint_summary"] = ep.get("summary")

    tm = json.load(open(os.path.join(d, "time_B_seed0.json")))
    out["velocity_mse_by_t"] = tm.get("velocity_mse_by_t")

    cond = json.load(open(os.path.join(d, "condition_B_seed0.json")))
    out["condition_per_scenario"] = cond.get("per_scenario")

    for name in ("fit_A_seed0_tiny-fit_A_seed0", "fit_B_seed0_tiny-fit_B_seed0",
                 "fit_A_seed0_tiny-fit_A_seed0_balanced", "fit_B_seed0_tiny-fit_B_seed0_balanced"):
        p = os.path.join(d, name + ".json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    main_tex, region_tex = multiclass_tables()
    with open(os.path.join(OUT, "tab_main.tex"), "w") as f:
        f.write(main_tex + "\n")
    with open(os.path.join(OUT, "tab_region.tex"), "w") as f:
        f.write(region_tex + "\n")
    with open(os.path.join(OUT, "tab_binary.tex"), "w") as f:
        f.write(binary_table() + "\n")
    with open(os.path.join(OUT, "tab_remedy.tex"), "w") as f:
        f.write(remedy_table() + "\n")
    print("已写出 tab_main.tex / tab_region.tex / tab_binary.tex / tab_remedy.tex")
    print("\n=== 主表（前 6 行预览）===")
    print("\n".join(main_tex.splitlines()[:14]))


if __name__ == "__main__":
    main()
