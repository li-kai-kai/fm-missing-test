"""Report only observed multiclass results, separately by split, scenario and region."""
import csv
import json
from pathlib import Path

from .config import load_config


def write_report(out, split="test", run_tag=""):
    out = Path(out)
    cfg = load_config(str(out / "config_preprocess.yaml"))
    root = out / "evaluation" / split / (run_tag or "main")
    summary_file = root / "summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"请先运行 run_eval.py：{summary_file}")
    with summary_file.open() as f:
        summary = list(csv.DictReader(f))
    boot = json.loads((root / "bootstrap.json").read_text())
    splits = json.loads((out / "splits.json").read_text())
    lines = ["# 随机缺一个模态的四分类脑肿瘤分割", "",
             f"评价子集：{split}；病例数：{len(splits[split])}；任务：{cfg.task}。", "",
             "训练逐样本等概率隐藏 T1、T2、T1ce 或 FLAIR；测试逐病例覆盖四种缺失模式。",
             "原始标签 0/1/2/4 对应背景、NCR/NET、ED、ET；内部类别为 0/1/2/3。",
             "WT=1+2+4，TC=1+4，ET=4（此处为原始标签）。预测 NIfTI 恢复原始标签值。", "",
             "A：非背景互斥类别平均 soft Dice + 四类 CE；B：四通道纯速度 MSE。",
             "默认每 2000 步验证，每例固定一种场景。按场景×区域的病例平均 Dice 等权平均选择 best；",
             "最终完整验证仅复核 best，不重新选择 checkpoint。实际配置见各运行 config.yaml。", "",
             "| 方法 | 种子 | 缺失场景 | 区域 | 病例数 | Dice 均值 | 病例标准差 | HD95 均值(mm) | HD95 失败数 | 两者均空 | 秒/预测 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append("| " + " | ".join(str(r[k]) if k not in ("dice_mean", "dice_std", "hd95_mean", "seconds_mean")
                     else f"{float(r[k]):.4f}" for k in
                     ("model", "seed", "scenario", "region", "n", "dice_mean", "dice_std", "hd95_mean",
                      "hd95_failures", "both_empty", "seconds_mean")) + " |")
    lines += ["", "| 种子 | 场景 | 区域 | 配对数 | Dice B−A | 病例 bootstrap 95% 区间 |",
              "|---|---|---|---|---|---|"]
    for key, r in sorted(boot.items()):
        seed, scenario, region = key.split("|")
        lines.append(f"| {seed} | {scenario} | {region} | {r['n']} | "
                     f"{r['mean_diff_B_minus_A']:.4f} | [{r['boot_lo']:.4f}, {r['boot_hi']:.4f}] |")
    lines += ["", "空区域规则：两者均空 Dice=1；仅一侧为空 Dice=0、HD95=NaN 并计入失败数。",
              "HD95 均值只包含可计算病例，必须结合失败数和两者均空数阅读；GT 空时召回率无定义。",
              "每个预测的运行时间在三个区域行中重复记录，不得相加为推理总耗时。",
              "bootstrap 在每个训练种子、缺失场景、区域内按患者配对重采样，不覆盖训练随机性；",
              "多个区域/场景结果是并列评价，区间未进行多重比较校正。", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines))
    return path
