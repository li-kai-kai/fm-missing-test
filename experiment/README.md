# 缺失模态下 Flow Matching 直接生成脑肿瘤分割 —— 实验说明

## 数据来源与格式

* 目录：`/SimCLR/data/BraTS2020`（本机路径）
* 结构：`BraTS20_Training_XXX/BraTS20_Training_XXX_{t1,t2,t1ce,flair,seg}.nii.gz`
* 可用病例：**369** 例带标注（`BraTS20_Training_*`）。
  另有 125 例 `BraTS20_Validation_*` **无标注**，不参与本实验。
* 审计结果见 `data_audit.json`：369 例全部为 240×240×155、1.0 mm 等体素、
  四模态与标注仿射一致、方向均为 LAS。因此**不做重采样、不做裁剪**，保留完整体积。

## 标签定义

`seg` 实际取值集合为 `{0, 1, 2, 4}`（见 `label_mapping.json`）：

| 原始值 | 含义 | 第一轮二分类 |
| :--- | :--- | :--- |
| 0 | 背景 | 背景 |
| 1 | NCR 坏死核心 | 肿瘤整体（前景） |
| 2 | ED 水肿 | 肿瘤整体（前景） |
| 4 | ET 增强肿瘤 | 肿瘤整体（前景） |

第一轮只做「肿瘤整体 vs 背景」二分类。子区域评价属于后续扩展阶段，
不能用二分类数字代替。

## 预处理

1. 统一到 RAS（`nibabel.as_closest_canonical`，确定性操作）。
2. 每个模态**在自身非零脑区内**做 z-score，脑区外置 0。
   不使用测试集全局统计量拟合。
3. **不裁剪**：保留完整 240×240×155 体积，因此不存在任何依赖
   真实肿瘤 mask 或被隐藏模态的裁剪决策。
4. 归一化**之后**才把被隐藏模态置零，存在标记同步更新。
5. 缓存为 `cache/<case>.npz`：`img` float16 `[4,D,H,W]`、`seg` uint8、`support` bool。
   `support`（四模态非零并集）**仅**用于训练期 patch 位置采样，
   绝不作为模型输入，也不用于推理。

## 三种输入场景

存在标记按 `[T1, T2, T1ce, FLAIR]` 排列：

| 场景 | 可用模态 | 存在标记 | 说明 |
| :--- | :--- | :--- | :--- |
| C0 | T1、T2、T1ce、FLAIR | `[1,1,1,1]` | 完整模态 |
| C1 | T1、T2、FLAIR | `[1,1,0,1]` | 缺 T1ce，**主要比较** |
| C2 | T2、FLAIR | `[0,1,0,1]` | 极端缺失 |

## 两个模型

| | A：普通分割 | B：FM 分割 |
| :--- | :--- | :--- |
| 输入通道 | 4 MRI + 4 存在标记 = 8 | 4 MRI + 4 存在标记 + 2 noisy mask + 1 时间通道 = 11 |
| 输出 | 2 通道分类 logits | 2 通道速度 |
| 损失 | 前景 soft Dice + 交叉熵（权重各 1） | 纯速度 MSE |
| 推理 | 滑窗融合连续分数后统一 argmax | 16 步 Euler，全体积统一更新 |

主干完全相同：4 级 3D U-Net，通道 `[16,32,64,128]`，GroupNorm。
实测参数量 A = 1,404,306，B = 1,405,602（差值 1,296 = 3 个额外输入通道 × 16 × 27），
主干宽度与深度一致。

## 运行方式

```bash
python3 run_preprocess.py --out experiment     # 审计 + 划分 + 预处理缓存
python3 run_sanity.py  --out experiment --steps 1500   # 阶段一
python3 run_train.py   --method A --seed 0     # 阶段二（A/B 各一次）
python3 run_train.py   --method B --seed 0
python3 run_eval.py    --out experiment --split test --methods A,B --seeds 0,1,2
python3 run_figures.py --out experiment
python3 run_case_figs.py --out experiment
python3 run_report.py  --out experiment
```

## 两个必须注意的实现要点

1. **数据 RNG 与模型噪声 RNG 分开**（`fmexp.data.step_rngs`）。
   两者都只依赖 `(seed, step)`，所以方法 A 与方法 B 在相同种子下看到的
   患者、patch、增强、缺失场景完全一致；B 额外抽噪声不会改变数据顺序。
2. **FM 推理时每一步所有滑窗必须读取同一份当前 y**，融合出完整速度场后再统一更新；
   同一步内不得边预测窗口边改 y。重叠区域共享同一份初始噪声。
   `run_sanity.py` 用 oracle 速度场与解析解两条独立路径验证了这一点。

## 复现性说明

* 划分种子 42，训练种子 0/1/2，初始噪声按「病例 ID + eval_seed」固定。
* 数据增强与场景采样由 `(seed, step)` 完全决定，可重放。
* GPU 上 cudnn 不保证逐比特可复现；同一配置重复运行可能有极小数值差异。
