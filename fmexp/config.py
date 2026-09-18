"""实验配置。

所有会改变实验结论的数值都集中在这里，并写入实验目录，便于复现。
对应方案文档 §4.3 / §2 / §5 的起始配置。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List

import yaml

# 模态顺序在全局固定：[T1, T2, T1ce, FLAIR]（方案 §3.1）
MODALITIES: tuple = ("t1", "t2", "t1ce", "flair")

# 三种输入场景，存在标记按 MODALITIES 顺序（方案 §3.2）
SCENARIOS = {
    "C0": {"avail": (1, 1, 1, 1), "name": "完整模态"},
    "C1": {"avail": (1, 1, 0, 1), "name": "缺 T1ce"},
    "C2": {"avail": (0, 1, 0, 1), "name": "T2+FLAIR"},
}
MISSING_ONE_SCENARIOS = [f"missing_{m}" for m in MODALITIES]
for i, m in enumerate(MODALITIES):
    SCENARIOS[f"missing_{m}"] = {
        "avail": tuple(int(j != i) for j in range(4)), "name": f"缺 {m}",
    }
TASKS = ("binary_wt", "multiclass_missing_one")
SCENARIO_ORDER: List[str] = ["C0", "C1", "C2"]
PRIMARY_SCENARIO = "C1"  # 预先指定的主要比较（方案 §1）


@dataclass
class Config:
    task: str = "binary_wt"
    # ---- 路径 ----
    data_root: str = "/SimCLR/data/BraTS2020"
    out_root: str = "experiment"

    # ---- 数据划分（方案 §2.1）----
    n_cases: int = 100
    n_train: int = 60
    n_val: int = 20
    n_test: int = 20
    split_seed: int = 42

    # ---- 预处理（方案 §2.3）----
    # 数据已是 1mm 等体素、已配准，因此不重采样、不裁剪，保留完整 240x240x155。
    # 保留全脑体积 = 不存在任何依赖肿瘤 mask / 被隐藏模态的裁剪决策。
    resample: bool = False
    target_spacing: tuple = (1.0, 1.0, 1.0)
    reorient_to: str = "RAS"

    # ---- patch 与训练（方案 §4.3）----
    patch: int = 64
    batch_size: int = 4          # 有效 batch size
    grad_accum: int = 1          # 显存不足时提高；有效 batch = batch_size * grad_accum
    tumor_center_prob: float = 0.5   # 50% 围绕肿瘤，50% 随机脑区
    # ---- 诊断专用开关；默认值 == 冻结行为，正式实验不受影响 ----
    augment: bool = True                 # False 时不施加任何增强（小样本拟合检查用）
    aug_keep_background_zero: bool = False   # True 时增强只在脑支撑内施加，背景保持严格 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 10_000      # 每个模型 10,000 次优化器更新
    val_every: int = 2_000
    val_mode: str = "balanced"  # 每例固定一个场景；full 可恢复每例全部场景
    seeds: tuple = (0, 1, 2)
    main_seed: int = 0

    # ---- 模型（方案 §3.1）----
    base_channels: int = 16
    channel_mult: tuple = (1, 2, 4, 8)   # -> [16, 32, 64, 128]
    num_levels: int = 4
    norm_groups: int = 8

    # ---- 推理（方案 §5）----
    sw_window: int = 64
    sw_overlap: float = 0.5      # 50% 重叠
    fm_steps: int = 16           # Euler 步数
    eval_seed: int = 1234        # 固定初始噪声用；同病例三场景共用

    # ---- 运行 ----
    device: str = "cuda"
    # bf16 autocast：实测本模型受显存/带宽限制，bf16 无明显加速（A 9.8s->8.2s，
    # B 24.8s->24.5s），而 FM 要连续积分 16 步，低精度会累积舍入。
    # 因此推理保持 fp32，只做纯速度语义上的保真。
    amp: bool = False
    num_workers: int = 0         # 体积常驻内存，无需 worker

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"未知 task: {self.task}")

    @property
    def n_classes(self):
        return 4 if self.task == "multiclass_missing_one" else 2

    @property
    def scenarios(self):
        return MISSING_ONE_SCENARIOS if self.n_classes == 4 else SCENARIO_ORDER


def _tuplize(d: dict) -> dict:
    for k, v in list(d.items()):
        if isinstance(v, list):
            d[k] = tuple(v)
    return d


def build_config(method: str, **overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"未知配置项: {k}")
        setattr(cfg, k, v)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), f, allow_unicode=True, sort_keys=False)


def load_config(path: str) -> Config:
    with open(path) as f:
        d = yaml.safe_load(f)
    return Config(**_tuplize(d or {}))
