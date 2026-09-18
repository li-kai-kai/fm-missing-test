"""把 diagnostics/*.json 渲染成中文 markdown 报告（风格对齐 fmexp/multiclass_report.py）。

脚本只负责数据表与可机器判定的预注册条目；人工结论写在
`<!-- 人工结论开始 -->` / `<!-- 人工结论结束 -->` 之间，重新生成时会被保留。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

REGIONS = ("WT", "TC", "ET")
BEGIN = "<!-- 人工结论开始 -->"
END = "<!-- 人工结论结束 -->"


def _f(v, n=4):
    if v is None:
        return "—"
    try:
        if v != v:
            return "—"
    except Exception:
        return "—"
    return f"{float(v):.{n}f}"


def table(headers: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def load_all(out: str) -> Dict[str, dict]:
    res = {}
    for p in sorted(glob.glob(os.path.join(out, "diagnostics", "*.json"))):
        with open(p) as f:
            res[os.path.basename(p)[:-5]] = json.load(f)
    return res


def _pick(data: Dict[str, dict], check: str) -> List[tuple]:
    return sorted((k, v) for k, v in data.items() if v.get("_check") == check)


# ---------------------------------------------------------------- 各节

def sec_location(L, data):
    items = _pick(data, "location")
    if not items:
        L.append("（缺少 location 产物：请先运行 `run_diagnostics.py --check location`）")
        return
    for name, d in items:
        L.append(f"### {name}")
        for mk, m in sorted(d["per_method"].items()):
            L.append("")
            L.append(f"**{mk}**")
            rows = []
            for scen, s in sorted(m["scenarios"].items()):
                rows.append([scen, f"{s['tp']:,}", f"{s['fn']:,}", f"{s['fp_in']:,}",
                             f"{s['fp_out']:,}", f"{s['fp_out_frac']:.1%}",
                             _f(s["recall"], 3), _f(s["precision"], 3),
                             f"{s['components_median']:.0f}",
                             _f(s.get("fp_out_within_10mm"), 2)])
            L += table(["缺失场景", "TP", "FN", "FP脑内", "FP脑外", "脑外占FP",
                        "召回", "精确", "碎块中位", "脑外FP≤10mm占比"], rows)
            L.append("")
        # 混淆矩阵只取第一个场景做示例
        first = sorted(m["scenarios"].items())[0][1] if m["scenarios"] else None
        if first and first.get("confusion"):
            import numpy as np
            c = np.array(first["confusion"], dtype=float)
            rs = c.sum(1, keepdims=True)
            p = np.divide(c, np.maximum(rs, 1))
            names = ["背景", "NCR", "ED", "ET"]
            rows = [[names[i]] + [_f(x, 4) for x in p[i]] + [f"{int(rs[i,0]):,}"]
                    for i in range(4)]
            L.append("混淆矩阵（行=真实，列=预测，最后一列为该真实类体素数）")
            L.append("")
            L += table(["真实\\预测"] + names + ["n"], rows)
            L.append("")


def sec_fit(L, data):
    items = _pick(data, "fit")
    if not items:
        L.append("（缺少 fit 产物：请先运行 `run_diagnostics.py --check fit --method B`）")
        return
    for name, d in items:
        L.append(f"### {name}（{d['n_train']} 训练例 / {d['n_test']} 测试例）")
        L.append("")
        rows = []
        for g in ("train", "test"):
            gd = d["groups"][g]
            for region in REGIONS:
                rows.append([g, region, _f(gd.get(region, {}).get("dice_mean")),
                             _f(gd.get(region, {}).get("dice_masked_mean"))])
        L += table(["病例组", "区域", "Dice", "掩蔽脑外后 Dice"], rows)
        L.append("")
        L.append(f"碎块中位数：训练 {d['groups']['train']['components_median']:.0f}，"
                 f"测试 {d['groups']['test']['components_median']:.0f}"
                 f"（掩蔽后 {d['groups']['test']['components_masked_median']:.0f}）")
        L.append("")


def sec_time(L, data):
    items = _pick(data, "time")
    if not items:
        L.append("（缺少 time 产物：请先运行 `run_diagnostics.py --check time --method B`）")
        return
    for name, d in items:
        L.append(f"### {name}")
        L.append("")
        L.append("**分时间速度 MSE**（patch 上沿理想路径；R² = 1 − MSE/Var(v)）")
        L.append("")
        rows = []
        for b in d["velocity_mse_by_t"]["bins"]:
            rows.append([_f(b["t"], 3), _f(b["mse_model"]), _f(b["var_v"]),
                         _f(b["r2"], 3), _f(b["mse_prior_only_ref"]),
                         _f(b["ratio_to_prior_only"], 3)])
        L += table(["t", "模型 MSE", "Var(v)", "R²", "仅先验参考", "模型/参考"], rows)
        L.append("")
        bins = d["velocity_mse_by_t"]["bins"]
        pc = bins[len(bins) // 2]["per_class"]
        # JSON 往返后类别键会变成字符串
        L.append(f"分类别 MSE（t={_f(bins[len(bins)//2]['t'],3)}）：" + "，".join(
            f"{n}={_f(pc[str(i)]['mse'])}" for i, n in enumerate(["背景", "NCR", "ED", "ET"])))
        L.append("")
        if d.get("trajectory_agg"):
            L.append("**理想路径 vs 实际轨迹**（同一批 t 上的速度 MSE 与各步 Dice）")
            L.append("")
            rows = []
            for r in d["trajectory_agg"]:
                rows.append([_f(r["t"], 3), _f(r["drift"], 4), _f(r["v_ideal"], 4),
                             _f(r["v_traj"], 4), _f(r["dice_ideal"], 3),
                             _f(r["dice_traj"], 3)])
            L += table(["t", "轨迹偏离理想", "MSE@理想点", "MSE@实际轨迹",
                        "Dice@理想点", "Dice@实际轨迹"], rows)
            L.append("")
            L.append(f"Var(v) = {_f(d.get('var_v'))}")
            L.append("")


def sec_condition(L, data):
    items = _pick(data, "condition")
    if not items:
        L.append("（缺少 condition 产物：请先运行 `run_diagnostics.py --check condition --method B`）")
        return
    for name, d in items:
        L.append(f"### {name}（{len(d['cases'])} 例）")
        L.append("")
        rows = []
        for scen, per in sorted(d["per_scenario"].items()):
            for variant in ("real", "zero", "swap_case", "shuffle_vox"):
                rows.append([scen, variant] + [_f(per.get(variant, {}).get(r)) for r in REGIONS])
        L += table(["缺失场景", "MRI 条件", "WT", "TC", "ET"], rows)
        L.append("")
        if d.get("noise_sweep_WT"):
            L.append(f"固定 MRI 变噪声：WT Dice 均值 {_f(sum(d['noise_sweep_WT'])/len(d['noise_sweep_WT']))}"
                     f"（各次 {', '.join(_f(x,3) for x in d['noise_sweep_WT'])}）")
            L.append(f"固定噪声变 MRI：WT Dice 均值 {_f(sum(d['mri_sweep_WT'])/len(d['mri_sweep_WT']))}"
                     f"（各次 {', '.join(_f(x,3) for x in d['mri_sweep_WT'])}）")
            L.append("")


def sec_steps(L, data):
    items = _pick(data, "steps")
    if not items:
        L.append("（缺少 steps 产物：请先运行 `run_diagnostics.py --check steps --method B`）")
        return
    for name, d in items:
        L.append(f"### {name}（{len(d['cases'])} 例）")
        L.append("")
        for scen, per in sorted(d["per_scenario"].items()):
            rows = []
            for s, v in sorted(per["summary"].items(), key=lambda kv: int(kv[0])):
                nc = v.get("n_components_median")
                rows.append([s] + [_f(v.get(r)) for r in REGIONS]
                            + ["—" if nc is None else f"{nc:.0f}"])
            L += table(["Euler 步数", "WT", "TC", "ET", "碎块中位"], rows)
            L.append("")


def sec_endpoint(L, data):
    items = _pick(data, "endpoint")
    if not items:
        L.append("（缺少 endpoint 产物：请先运行 `run_diagnostics.py --check endpoint --method B`）")
        return
    for name, d in items:
        L.append(f"### {name}（{len(d['cases'])} 例）")
        L.append("")
        L.append("终点**连续**状态的置信间隔（argmax − 次大值；one-hot 目标下满分是 1.0）。"
                 "间隔接近 0 意味着该体素的判定基本是随机翻转。")
        L.append("")
        rows = []
        for s, m in sorted(d["summary"].items(), key=lambda kv: int(kv[0])):
            rows.append([s, _f(m.get("dice_WT"), 3), _f(m.get("margin_median"), 3),
                         _f(m.get("frac_margin_lt_0.25"), 3),
                         _f(m.get("margin_median_in_brain"), 3),
                         _f(m.get("margin_median_out_brain"), 3),
                         _f(m.get("channelsum_mean"), 3)])
        L += table(["Euler 步数", "WT Dice", "间隔中位", "间隔<0.25 占比",
                    "脑内间隔中位", "脑外间隔中位", "通道和"], rows)
        L.append("")


def sec_evaltest(L, data):
    items = _pick(data, "evaltest")
    if not items:
        L.append("（缺少 evaltest 产物：请先运行 `run_diagnostics.py --check evaltest`）")
        return
    def _tag(d):
        return (os.path.basename(str(d.get("run_dir") or "").rstrip("/"))
                or f"{d.get('method', '?')}_seed{d.get('seed', '?')} (原始)")

    rows = []
    for name, d in items:
        rows.append([_tag(d)] + [_f(d["overall"].get(r)) for r in REGIONS]
                    + [str(len(d["cases"]))])
    if rows:
        L += table(["run", "WT", "TC", "ET", "病例数"], rows)
        L.append("")
    for name, d in items:
        tag = _tag(d)
        rows = []
        for scen, per in sorted(d["per_scenario"].items()):
            rows.append([scen] + [_f(per.get(r)) for r in REGIONS]
                        + ["—" if per.get("components_median") is None
                           else f"{per['components_median']:.0f}"])
        L.append(f"**{tag}** 按缺失场景")
        L.append("")
        L += table(["缺失场景", "WT", "TC", "ET", "碎块中位"], rows)
        L.append("")


def sec_training(L, data, out: str):
    """延长训练与背景置零变体的训练曲线（读 diagnostics/runs/*/log.jsonl）。"""
    runs = sorted(glob.glob(os.path.join(out, "diagnostics", "runs", "*")))
    if not runs:
        L.append("（缺少诊断重训目录 diagnostics/runs/）")
        return
    for rd in runs:
        log = os.path.join(rd, "log.jsonl")
        if not os.path.exists(log):
            continue
        L.append(f"### {os.path.basename(rd)}")
        L.append("")
        rows, train = [], []
        with open(log) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "val":
                    rows.append([r["step"]] + [_f(r.get(f"dice_{s}")) for s in
                                               ("missing_t1", "missing_t2",
                                                "missing_t1ce", "missing_flair")]
                                + [_f(r["score"])])
                elif r["type"] == "train" and r["step"] % 1000 == 0:
                    train.append((r["step"], r.get("mse", r.get("loss"))))
        if rows:
            L += table(["step", "缺T1", "缺T2", "缺T1ce", "缺FLAIR", "score"], rows)
            L.append("")
        if train:
            L.append("训练损失（每 1000 步）：" + "，".join(f"{s}:{_f(v,4)}" for s, v in train))
            L.append("")


def sec_criteria(L, data):
    """可机器判定的预注册条目；结论列直接写判定，避免「符合/不符合」指向不明。"""
    L.append("| 判读条目 | 阈值 | 实测 | 判定 |")
    L.append("|---|---|---|---|")

    def row(name, thr, val, verdict):
        L.append(f"| {name} | {thr} | {val} | {verdict} |")

    # 只取主实验的拟合对照（8 训练例）；tiny-fit 的产物是各方法自己的训练例/测试例，
    # 语义不同，混进来会把「泛化差距」算错。
    fit = [(k, v) for k, v in _pick(data, "fit") if "tiny-fit" not in k and v.get("n_train", 0) >= 5]
    if fit:
        d = fit[0][1]
        tr = d["groups"]["train"]["WT"]["dice_mean"]
        # 测试一侧优先用完整测试划分的 evaltest：check_fit 只跑 5 例，是任意抽样，
        # 拿它算泛化差距会把抽样噪声读成泛化差距（曾出现 0.131 对真实 0.058）。
        full = [v["overall"]["WT"] for k, v in _pick(data, "evaltest")
                if v.get("method") == d.get("method")
                and "diagnostics" not in str(v.get("run_dir") or "")]
        if full:
            te, note = full[0], "完整测试划分"
        else:
            te, note = d["groups"]["test"]["WT"]["dice_mean"], f"{d['n_test']} 例子集"
        gap = tr - te
        row(f"训练例 − 测试例 WT Dice（测试侧：{note}）",
            "|差距| < 0.10 判为非泛化问题", _f(gap, 4),
            "差距小 → 不是泛化问题" if abs(gap) < 0.10 else "差距大 → 泛化问题")
    cond = _pick(data, "condition")
    if cond:
        d = cond[0][1]
        for scen, per in sorted(d["per_scenario"].items()):
            drop = per["real"]["WT"] - per["shuffle_vox"]["WT"]
            row(f"{scen}：真实 − 打乱 MRI 的 WT Dice", "|变化| ≥ 0.05 判为真正条件化",
                _f(drop, 4),
                "真正用了 MRI" if abs(drop) >= 0.05 else "影响微弱 → 近似无条件")
    st = _pick(data, "steps")
    if st:
        d = st[0][1]
        for scen, per in sorted(d["per_scenario"].items()):
            s = per["summary"]
            if "16" in s and "64" in s:
                delta = s["64"]["WT"] - s["16"]["WT"]
                if abs(delta) < 0.02:
                    verdict = "积分不是瓶颈"
                elif delta < 0:
                    verdict = "步数越多越差 → 是场本身的问题，加步数无法补救"
                else:
                    verdict = "步数不足 → 数值积分是瓶颈"
                row(f"{scen}：16→64 步 WT Dice 变化", "|变化| < 0.02 判为积分非瓶颈",
                    _f(delta, 4), verdict)


def preserve_manual(path: str) -> str:
    if not os.path.exists(path):
        return "_（待填）_"
    txt = open(path).read()
    if BEGIN in txt and END in txt:
        return txt.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    return "_（待填）_"


def write_diag_report(out: str) -> str:
    data = load_all(out)
    path = os.path.join(out, "diagnostics", "report.md")
    manual = preserve_manual(path)
    L: List[str] = []
    L.append("# 四分类缺失模态 FM 失败归因诊断")
    L.append("")
    L.append("诊断脚本 `run_diagnostics.py` 的只读测量结果。所有产物位于 "
             "`experiment_multiclass/diagnostics/`，与主实验的 `runs/`、`evaluation/`、"
             "`predictions/` 相互隔离。")
    L.append("")
    L.append("诊断性推理变体（按脑支撑掩蔽、打乱/置零 MRI）只作为读数，**不是方法结果**，"
             "不得与主实验的 A/B 比较混用。")
    L.append("")
    if data:
        L.append("已生成的产物：" + "，".join(f"`{k}`" for k in sorted(data)))
    else:
        L.append("尚无产物。")
    L.append("")

    L.append("## 1. 检查 1：小样本 / 训练病例完整生成过拟合")
    L.append("")
    sec_fit(L, data)
    L.append("## 2. 检查 2：时间分段与 MRI 条件")
    L.append("")
    sec_time(L, data)
    sec_condition(L, data)
    sec_endpoint(L, data)
    L.append("## 3. 检查 3：错误位置")
    L.append("")
    sec_location(L, data)
    L.append("### 3b. 背景置零增强变体与延长训练")
    L.append("")
    sec_training(L, data, out)
    L.append("## 3c. 各诊断 checkpoint 在完整测试划分上的评估")
    L.append("")
    sec_evaltest(L, data)
    L.append("## 4. 检查 4：步数扫描")
    L.append("")
    sec_steps(L, data)
    L.append("## 5. 预注册判读对照")
    L.append("")
    sec_criteria(L, data)
    L.append("")
    L.append("## 6. 人工结论")
    L.append("")
    L.append(BEGIN)
    L.append(manual)
    L.append(END)
    L.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path
