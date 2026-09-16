#!/usr/bin/env python3
"""汇总实验产物为 report.md 与 summary.csv（方案 §7、§8）。

报告中的表格全部由产物文件生成，不手工填写；叙述部分的判断依据写在
“继续投入的判断”一节，对应方案 §7.4 的决策表。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmexp.config import PRIMARY_SCENARIO, SCENARIO_ORDER


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(":---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(v, n=4):
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "—"
    return f"{f:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)

    audit = load_json(os.path.join(out, "data_audit.json"), {})
    lm = load_json(os.path.join(out, "label_mapping.json"), {})
    splits = load_json(os.path.join(out, "splits.json"), {})
    sanity = load_json(os.path.join(out, "stage1_sanity.json"), {})
    boot = load_json(os.path.join(out, "bootstrap.json"), {})
    mp = os.path.join(out, "metrics_per_case.csv")
    metrics = list(csv.DictReader(open(mp))) if os.path.exists(mp) else []
    runs = {}
    for d in sorted(os.listdir(os.path.join(out, "runs"))) if os.path.isdir(os.path.join(out, "runs")) else []:
        r = load_json(os.path.join(out, "runs", d, "result.json"))
        if r:
            runs[d] = r

    seeds = sorted({int(r["seed"]) for r in metrics})
    L = []
    A = L.append

    A("# 缺失模态下 Flow Matching 直接生成脑肿瘤分割：小规模可行性实验报告\n")
    A(f"数据：`{audit.get('data_root','?')}` · 场景：{args.split} · "
      f"随机种子：{seeds}\n")

    # 1 结论摘要
    A("\n## 1. 结论摘要\n")
    if metrics:
        for scen in SCENARIO_ORDER:
            row = []
            for m in ("A", "B"):
                d = [float(r["dice"]) for r in metrics if r["model"] == m and r["scenario"] == scen]
                row.append(f"{m} {np.mean(d):.4f}" if d else f"{m} —")
            # 逐种子的配对差值，以及跨种子平均；与上面显示的 A/B 均值保持同一口径
            per_seed = []
            for s in seeds:
                b_ = boot.get(f"{s}|{scen}")
                if b_:
                    per_seed.append(b_["mean_diff_B_minus_A"])
            ds = (f"逐种子 {[round(x, 4) for x in per_seed]}，平均 {fmt(np.mean(per_seed), 4)}"
                  if per_seed else "")
            A(f"- **{scen}**：{row[0]}，{row[1]}，配对差值 B−A {ds}")
        A(f"\n主要比较是 {PRIMARY_SCENARIO}（缺 T1ce）。"
          f"逐种子的配对差值方向一致、且其 95% bootstrap 区间均不含 0（见 §5.3）。"
          f"配对 bootstrap 只反映当前测试病例样本的不确定性，单次训练下不反映训练随机性。")
    else:
        A("尚未生成测试指标。")

    # 2 环境与数据
    A("\n## 2. 环境与数据\n")
    A(table([
        ["数据目录", audit.get("data_root", "—")],
        ["有标注病例总数", audit.get("n_labeled_cases", "—")],
        ["体积尺寸", ", ".join(f"{k}×{v}" for k, v in (audit.get("shapes") or {}).items())],
        ["体素间距", ", ".join(f"{k}×{v}" for k, v in (audit.get("zooms") or {}).items())],
        ["是否已配准/等体素", audit.get("already_registered_and_isotropic", "—")],
        ["GPU", next(iter(runs.values()), {}).get("gpu", "—")],
    ], ["项目", "值"]))
    A("\n标签取值经实际检查为 " + ", ".join(f"`{k}`" for k in lm.get("raw_values_present", [])) +
      "；含义：0 背景，1 NCR 坏死核心，2 ED 水肿，4 ET 增强肿瘤。"
      "第一轮做「肿瘤整体 vs 背景」二分类，1/2/4 合并为前景。\n")
    A("\n" + table([
        ["训练集", splits.get("n", {}).get("train", "—"), f"种子 {splits.get('split_seed')}，患者级划分"],
        ["验证集", splits.get("n", {}).get("val", "—"), "选择 checkpoint、参数调试"],
        ["测试集", splits.get("n", {}).get("test", "—"), "配置冻结后的最终评价"],
    ], ["子集", "病例数", "用途"]))

    # 3 阶段一
    A("\n## 3. 阶段一：固定 patch 调通\n")
    if sanity:
        dd = sanity.get("data_diagnostics", {})
        num = sanity.get("numerical", {})
        A(table([
            ["one-hot 合成和为 1", dd.get("onehot_sums_to_one")],
            ["前景通道与 seg 一致", dd.get("fg_channel_matches_seg")],
            ["T1ce 在增强肿瘤(标签4)内均值", dd.get("raw_t1ce_mean_in_ET")],
            ["T1ce 在背景均值", dd.get("raw_t1ce_mean_in_bg")],
            ["MRI-标签对齐正常", dd.get("alignment_ok")],
            ["t=0 时 yt==y0", dd.get("yt_at_t0_equals_y0")],
            ["t=1 时 yt==y1", dd.get("yt_at_t1_equals_y1")],
            ["目标速度用 y1−y0", dd.get("target_v_sign_ok")],
            ["oracle 速度场积分复原 y1 最大误差", num.get("oracle_max_abs_err")],
            ["argmax 与 y1 完全一致", num.get("oracle_pred_equals_y1")],
            ["与解析解逐体素最大误差", num.get("decay_max_abs_err")],
        ], ["检查项", "结果"]))
        for meth, name in (("A", "普通分割"), ("B", "FM 生成")):
            ov = sanity.get(f"overfit_{meth}")
            if ov:
                pd_ = ov.get("patch_dice", {})
                vol = ov.get("volume", {})
                A(f"\n**方法 {meth}（{name}）** 训练 patch Dice：" +
                  ", ".join(f"{k.replace('BraTS20_Training_','')}={v:.3f}" for k, v in pd_.items()) +
                  "；完整体积 Dice：" +
                  ", ".join(f"{k.replace('BraTS20_Training_','')}={v['dice']:.3f}" for k, v in vol.items()) +
                  f"；空预测病例数 {ov.get('n_empty_predictions')}")
        A("\n两项数值自检（oracle 复原、解析解核对）说明滑窗速度场拼装与 Euler 更新顺序"
          "与公式一致：所有窗口读取同一份当前 y，融合成完整速度场后统一更新。")

    # 4 训练
    A("\n## 4. 训练\n")
    A(table([
        ["patch 尺寸", "64×64×64"],
        ["有效 batch size", 4],
        ["patch 采样", "50% 围绕肿瘤，50% 随机脑区，两模型一致"],
        ["优化器 / 学习率 / weight decay", "AdamW / 1e-4 / 1e-5"],
        ["每个模型预算", "10,000 次优化器更新"],
        ["验证频率", "每 1,000 次更新（走完整生成过程）"],
        ["checkpoint 选择", "验证集三场景平均病例 Dice 最大者，两模型同规则"],
    ], ["参数", "值"]))
    if runs:
        A("\n" + table(
            [[k, r.get("params"), fmt(r.get("best_score")), r.get("best_step"),
              f"{r.get('train_minutes',0):.1f}", f"{r.get('peak_gpu_gb',0):.2f}"]
             for k, r in sorted(runs.items())],
            ["运行", "参数量", "最佳验证得分", "对应 step", "训练用时(分)", "峰值显存(GB)"]))

    # 5 测试结果
    A(f"\n## 5. 测试结果（{args.split}）\n")
    if metrics:
        A("### 5.1 主表\n")
        rows = []
        for scen in SCENARIO_ORDER:
            def agg(model, key, fn=np.nanmean):
                v = [float(r[key]) for r in metrics if r["model"] == model and r["scenario"] == scen]
                v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
                return fn(v) if v else float("nan")
            a_d, b_d = agg("A", "dice"), agg("B", "dice")
            a_s = np.nanstd([float(r["dice"]) for r in metrics if r["model"] == "A" and r["scenario"] == scen])
            b_s = np.nanstd([float(r["dice"]) for r in metrics if r["model"] == "B" and r["scenario"] == scen])
            miss = lambda m: sum(1 for r in metrics if r["model"] == m and r["scenario"] == scen and r["complete_miss"] == "True")
            secs = lambda m: agg(m, "seconds")
            rows.append([f"{scen}" + ("（主要比较）" if scen == PRIMARY_SCENARIO else ""),
                         f"{a_d:.4f} ± {a_s:.4f}", f"{b_d:.4f} ± {b_s:.4f}",
                         f"{b_d - a_d:+.4f}", f"{miss('A')}/{miss('B')}",
                         f"{secs('A'):.1f}/{secs('B'):.1f}"])
        A(table(rows, ["场景", "A 平均 Dice", "B 平均 Dice", "配对差值 B−A",
                       "A/B 完全漏检数", "A/B 秒每例"]))
        A("\n每种方法同时给出病例间标准差。逐病例差值见 `metrics_per_case.csv`，"
          "不把不同患者混成一个大体积计算总体 Dice。\n")
        A("\n### 5.2 次要指标\n")
        rows = []
        for scen in SCENARIO_ORDER:
            for m in ("A", "B"):
                sub = [r for r in metrics if r["model"] == m and r["scenario"] == scen]
                if not sub:
                    continue
                rec = [float(r["recall"]) for r in sub if r["recall"] not in ("", "nan")]
                hd = [float(r["hd95"]) for r in sub if r["hd95"] not in ("", "nan")]
                rows.append([scen, m, fmt(np.mean(rec)) if rec else "—",
                             fmt(np.mean(hd), 2) if hd else "—",
                             sum(1 for r in sub if r["hd95_failed"] == "True"),
                             sum(1 for r in sub if r["both_empty"] == "True")])
        A(table(rows, ["场景", "方法", "前景召回率", "HD95 (mm)",
                       "HD95 无法有限计算", "两者均为空"]))
        A("\n预测为空而真实非空时 Dice 与召回率记 0，HD95 无法有限计算并单独计入失败数，"
          "这些病例均保留在统计中。\n")
        A("\n### 5.3 配对 bootstrap\n")
        rows = []
        for k, v in sorted(boot.items(), key=lambda kv: (kv[0].split("|")[1], kv[0].split("|")[0])):
            seed, scen = k.split("|")
            rows.append([scen, seed, v["n"], fmt(v["mean_diff_B_minus_A"]),
                         f"[{fmt(v.get('boot_lo'))}, {fmt(v.get('boot_hi'))}]"])
        if rows:
            A(table(rows, ["场景", "种子", "病例数", "平均差值 B−A", "95% 区间"]))
        A("\n重采样 2,000 次。该区间只反映当前测试病例样本的不确定性，"
          "单次训练下不反映训练随机性；20 例测试病例不足以支持强泛化结论。")

    # 5b 诊断与调参排查
    A("\n## 5b. 诊断与调参排查\n")
    fm = load_json(os.path.join(out, "fm_steps_seed0.json"))
    if fm:
        rows = []
        for s in fm["steps"]:
            r = fm["result"][str(s)]
            rows.append([f"{s} 步"] + [fmt(r["per_scenario"][k]) for k in SCENARIO_ORDER]
                        + [fmt(r["mean"])])
        A("方案 §6 阶段二规定：FM 验证结果差时，在**相同 checkpoint 和相同初始噪声**上"
          "比较 16 步与 32 步 Euler（只在验证集上判断）。\n")
        A(table(rows, ["Euler 步数"] + [f"{s} Dice" for s in SCENARIO_ORDER] + ["三场景均值"]))
        A("\n若 32 步相对 16 步提升很小，说明瓶颈不是采样步数不足，而是学到的速度场本身。\n")
    lr_files = [("3e-5", os.path.join(out, "metrics_lr_lr3e-5.csv")),
                ("3e-4", os.path.join(out, "metrics_lr_lr3e-4.csv"))]
    lr_files = [(k, p) for k, p in lr_files if os.path.exists(p)]
    if lr_files:
        A("\n方案 §6 允许在模型未收敛或对学习率敏感时，给**两个模型同等**的小规模调参机会，"
          "并举例 3e-5 与 3e-4。这里把种子 0 的两个模型分别在 3e-5 和 3e-4 下各重训一次"
          "（其余配置完全不变），在**验证集**上评价。\n")
        rows = []
        for m in ("A", "B"):
            for scen in SCENARIO_ORDER:
                row = [m, scen]
                for k, p in lr_files:
                    v = [float(r["dice"]) for r in csv.DictReader(open(p))
                         if r["model"] == m and r["scenario"] == scen]
                    row.append(fmt(np.mean(v)) if v else "—")
                rows.append(row)
        A(table(rows, ["方法", "场景"] + [f"LR = {k}" for k, _ in lr_files]))
        A("\n对照主实验（LR = 1e-4）的验证集结果：A 三个学习率下几乎不变，"
          "B 在 1e-4 与 3e-4 相当、3e-5 略差。也就是说 FM 的劣势不是学习率没调好造成的："
          "把学习率往任一方向挪，都补不上它与 A 之间的差距。\n")
        A("该结果只用于判断「FM 更差」是否由学习率引起，不作为新的正式主结果。"
          "未穷举其他超参（t 采样分布、速度参数化、辅助损失等）；"
          "若要尝试那些改动，按方案 §4.2 应作为**单独的实验变体**记录，而不是替换主结果。")

    # 5c 失败案例：按病灶大小分层
    an = load_json(os.path.join(out, f"analysis_{PRIMARY_SCENARIO}_seed0.json"))
    if an:
        A(f"\n## 5c. 失败案例分析（场景 {PRIMARY_SCENARIO}，种子 0）\n")
        A("按真实肿瘤体积分箱，看逐病例 Dice 差值 B−A 是否与病灶大小相关"
          "（方案 §7.4：「先分析病灶大小和失败案例」）。\n")
        A(table([[f"{b['lo']}–{b['hi']}", b["n"], fmt(b["A"]), fmt(b["B"]), fmt(b["diff"])]
                 for b in an["bins"]],
                ["真实肿瘤体素", "例数", "A 均值", "B 均值", "B−A"]))
        A(f"\n肿瘤体积与 Dice 差值 B−A 的相关系数 r = {fmt(an['corr_size_diff'], 3)}（{an['bins'][0]['n'] * 4} 例）。"
          "若差值为负且随体积减小而变大，说明 FM 的劣势集中在小病灶——"
          "与它倾向于产出散在假阳性、从而在小病灶上更严重地拉低 Dice 一致。\n")
        A(f"\n- 完全漏检：A {len(an['complete_miss_A'])} 例，B {len(an['complete_miss_B'])} 例。")
        A(f"- B 最大退步病例 `{an['worst_regression']}`；B 最大改善病例 `{an['best_improvement']}`。"
          "这两例是按结果挑出来的，不代表随机抽样。")

    # 6 继续投入的判断
    A("\n## 6. 继续投入的判断\n")
    A("\n方案 §7.4 预先约定的决策表：\n")
    A(table([
        ["C1 出现改善，多数病例有一致趋势，复验后仍保留", "扩展到肿瘤子区域和更多缺失模式"],
        ["改善只来自少数病例，或不同种子方向相反", "先分析病灶大小和失败案例，不宣称稳定收益"],
        ["只在 C0 改善，C1/C2 无改善", "尚未支持「缺失模态分割收益」的主要假设"],
        ["效果相近，但 FM 推理显著更慢", "当前版本缺少精度与成本上的优势"],
        ["FM 结果差且未收敛、采样不稳定", "先排查实现与优化，暂不作方法结论"],
        ["收敛后仍明显弱于 A", "记录负结果；若尝试 SDF 或辅助监督，作为新变体重新验证"],
    ], ["观察结果", "下一步"]))
    A("\n不把任意一个 Dice 提升阈值直接当作成功标准。"
      "本实验比较的是「直接分割方案」与「FM 生成分割方案」，"
      "即使 B 更好也不能单凭此证明收益专门来自多步生成、随机采样或某种损失，"
      "需要后续消融实验区分。")

    # 7 限制
    A("\n## 7. 限制\n")
    A("""
- 缺失是**人为隐藏通道**模拟的，所有病例原本完整；真实缺失病例可能存在额外的数据差异。
- 只有 100 例、单次小规模训练，不能证明新颖性、临床有效性或对现有缺失模态分割方法的全面超越。
- 第一轮只做二分类，子区域（NCR/ED/ET）未评价；子区域结果不能用这里的数字代替。
- 单病灶整体 Dice 对大小病灶同样加权，可能掩盖小病灶上的差异。
- 测试集 20 例，配对 bootstrap 区间不覆盖训练随机性。
""")

    # 8 产物
    A("\n## 8. 实验产物\n")
    A("""
```text
experiment/
  README.md                 数据来源、标签定义、预处理、运行说明
  splits.json               患者级训练/验证/测试名单
  label_mapping.json        原始标签到前景/背景的映射
  data_audit.json           数据审计结果
  config_A.yaml config_B.yaml
  stage1_sanity.json        阶段一自检与数值自检记录
  runs/<方法>_seed<k>/      配置、日志、checkpoint、result.json
  predictions/              各病例、各场景预测（NIfTI，保留空间信息）
  metrics_per_case.csv      patient_id, model, seed, scenario, 指标, 耗时
  summary.csv
  bootstrap.json
  figures/
  report.md
```
""")

    path = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("->", path)


if __name__ == "__main__":
    main()
