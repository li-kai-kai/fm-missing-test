# 随机缺一个 MRI 模态、预测全部肿瘤标签

这是独立于 `experiment/` 的新实验。旧 WT 二分类的 checkpoint 不能用于四分类。

## 预先固定的设置

| 项目 | 设置 |
| --- | --- |
| 数据划分 | 复用 `experiment/splits.json` 的 60/20/20 患者划分 |
| 每个训练样本 | 等概率隐藏 T1、T2、T1ce、FLAIR 中恰好一个；增强后缺失通道仍严格为 0 |
| 存在标记 | 4 通道，顺序 T1、T2、T1ce、FLAIR |
| 四个互斥类别 | 背景、坏死及非增强核心 NCR/NET、水肿 ED、增强肿瘤 ET |
| 内部标签 | 原始 0/1/2/4 → 类别 0/1/2/3；预测 NIfTI 恢复 0/1/2/4 |
| A | 8 输入通道、4 输出 logits；非背景类别平均 soft Dice + 四类 CE |
| B | 13 输入通道（8 条件+4 noisy mask+1 时间）、4 输出速度；纯速度 MSE |
| FM 路径 | 四通道标准高斯 → 四通道 one-hot；16 步 Euler，最后统一 argmax |
| 预算 | 各 10000 次更新，batch=4，patch=64³，AdamW，LR=1e-4 |
| 训练期验证 | 每 2000 步，20 例各固定一种缺失模式，四种模式各 5 例 |
| 选择 best | 对每种场景×区域计算病例平均 Dice，再对 4×3 组等权平均 |
| 完整验证 | 选定 best 后评估 20 例×4 模式×3 区域，不重新选择 checkpoint |
| 测试 | 每例逐一隐藏四种模态，分别评价 WT、TC、ET 的 Dice、召回率、HD95 |

每例只有一个预测体积，WT/TC/ET 从该预测导出，三者不是互斥输出通道。
按原始标签定义 WT={1,2,4}、TC={1,4}、ET={4}。
本设置不采样完整模态，也不采样同时缺两个/三个模态。

训练期 step 0 每种缺失模式检查一例，不参与 checkpoint 选择。
默认训练总共生成 4+5×20+20×4=184 个验证预测；指标记录行数为其三倍，
不应把三行重复的推理耗时相加。A/B 使用相同数据 RNG 和固定验证分配。

## 运行

在仓库根目录执行。先做短程调通（真实训练病例，默认仅 10 次更新，不评估性能）：

```bash
python3 run_multiclass_sanity.py
```

准备独立缓存，复用原划分：

```bash
python3 run_preprocess.py --task multiclass_missing_one --out experiment_multiclass --splits-from experiment/splits.json
```

正式训练读取该目录的 `config_preprocess.yaml` 自动识别四分类任务：

```bash
python3 run_train.py --out experiment_multiclass --method A --seed 0
python3 run_train.py --out experiment_multiclass --method B --seed 0
```

按相同命令将 `--seed` 改为 1、2 复验。配置确定后运行测试和生成报告：

```bash
python3 run_eval.py --out experiment_multiclass --split test --methods A,B --seeds 0,1,2
python3 run_report.py --out experiment_multiclass --split test
```

只训练 seed 0 时，评估也使用 `--seeds 0`。验证集评估改为 `--split val`。
评估加载每次运行的实际配置；验证和测试指标分别保存在 `evaluation/val/main/`、
`evaluation/test/main/`。调参使用 `--run-tag`，另存子目录。
原有 `run_figures.py`、`run_case_figs.py`、`run_sanity.py` 是旧二分类专用脚本，
本实验使用上述调通和报告入口。

## 指标约定与产物

* 两者均空：Dice=1，召回率与 HD95 为 NaN，单独统计 `both_empty`。
* 仅一侧为空：Dice=0，HD95 为 NaN 并计入 `hd95_failed`；真实非空而预测为空还计入完全漏检。
* 真实区域为空时召回率无定义。ET 缺失可能出现，不能悄悄删除这些病例。
* HD95 沿用本仓库实现：合并两个方向的表面距离取 95 百分位，1 mm 体素间距。
  均值仅覆盖可计算病例，同时报告失败数；不声称等同于 BraTS 官方评分实现。
* 按患者、模型、训练种子、缺失模式、区域保存指标，分别汇总与配对 bootstrap。
  bootstrap 不覆盖训练随机性，12 个场景—区域组合未作多重比较校正。
* `runs/` 保存配置、模型、日志、完整验证明细；`evaluation/<split>/main/` 保存 CSV、
  bootstrap 和报告；`predictions/<split>/` 保存原始标签值的预测 NIfTI。
* 缓存和 checkpoint、预测体积由 `.gitignore` 排除；旧实验结果不作为本实验结果。

自动化检查：`python3 -m unittest discover -s tests -v`。
