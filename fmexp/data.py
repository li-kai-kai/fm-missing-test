"""数据审计、预处理、划分与 patch 采样。

关键防泄漏约定（方案 §2.3）：
  * 归一化统计量只从该模态自身脑区计算，不使用测试集全局统计量；
  * 不做依赖肿瘤 mask 的裁剪（保留完整体积）；
  * 归一化之后再把被隐藏模态置零，存在标记同步更新；
  * 训练可用真实 mask 采样含肿瘤 patch；验证/测试覆盖完整脑体积。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np

from .config import MODALITIES, SCENARIOS, SCENARIO_ORDER

# 原始标签 -> 前景/背景 的映射（方案 §2.2）。
# 经实际检查：BraTS2020 seg 取值集合为 {0,1,2,4}，
#   0 = 背景, 1 = NCR 坏死核心, 2 = ED 水肿, 4 = ET 增强肿瘤。
# 第一轮只做“肿瘤整体 vs 背景”二分类，故 1/2/4 合并为前景。
LABEL_MAPPING = {
    "dataset": "BraTS2020",
    "raw_values_present": [0, 1, 2, 4],
    "raw_meaning": {"0": "背景", "1": "NCR 坏死核心", "2": "ED 水肿", "4": "ET 增强肿瘤"},
    "binary": {"0": "背景", "1,2,4": "肿瘤整体（前景）"},
    "foreground": [1, 2, 4],
    "ignored": [],
    "note": "第一轮不做子区域；子区域评价属于后续扩展阶段。",
}

MULTICLASS_LABEL_MAPPING = {
    "dataset": "BraTS2020", "raw_values_present": [0, 1, 2, 4],
    "raw_to_class": {"0": 0, "1": 1, "2": 2, "4": 3},
    "class_to_raw": [0, 1, 2, 4],
    "classes": ["background", "NCR/NET", "ED", "ET"],
    "regions_internal": {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]},
}


def encode_segmentation(seg_raw, task="binary_wt"):
    if not np.isin(seg_raw, [0, 1, 2, 4]).all():
        raise ValueError("原始标签必须来自 {0,1,2,4}")
    if task == "binary_wt":
        return (seg_raw > 0).astype(np.uint8)
    if task != "multiclass_missing_one":
        raise ValueError(task)
    return np.where(seg_raw == 4, 3, seg_raw).astype(np.uint8)


def decode_segmentation(seg, n_classes):
    """保存 NIfTI 时恢复 BraTS 原始标签值；二分类保持 0/1。"""
    return np.asarray([0, 1, 2, 4], dtype=np.uint8)[seg] if n_classes == 4 else seg


# ---------------------------------------------------------------- 病例发现

def case_id(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def all_labeled_cases(data_root: str) -> List[str]:
    """有 seg 标注的病例（BraTS20_Training_*）。Validation 组无标注，不参与。"""
    ids = []
    for d in sorted(glob.glob(os.path.join(data_root, "BraTS20_Training_*"))):
        if not os.path.isdir(d):
            continue
        cid = case_id(d)
        if all(os.path.exists(os.path.join(d, f"{cid}_{m}.nii.gz")) for m in MODALITIES + ("seg",)):
            ids.append(cid)
    return ids


def select_cases(data_root: str, n_cases: int, seed: int) -> List[str]:
    """从合格病例中挑 n_cases 个。合格 = 四模态 + 标注齐全；无标签值异常检查在此完成。"""
    ids = all_labeled_cases(data_root)
    if len(ids) < n_cases:
        raise RuntimeError(f"合格病例不足：需要 {n_cases}，只有 {len(ids)}")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(ids), size=n_cases, replace=False))
    return [ids[i] for i in pick]


def make_splits(cases: Sequence[str], n_train: int, n_val: int, n_test: int, seed: int) -> Dict[str, List[str]]:
    """按患者划分。BraTS 每个病例 ID 即一位患者，天然满足患者级划分。"""
    assert n_train + n_val + n_test == len(cases)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cases))
    tr = [cases[i] for i in sorted(perm[:n_train])]
    va = [cases[i] for i in sorted(perm[n_train:n_train + n_val])]
    te = [cases[i] for i in sorted(perm[n_train + n_val:])]
    return {"train": tr, "val": va, "test": te}


# ---------------------------------------------------------------- 预处理

def _zscore_brain(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按该模态自身非零脑区做 z-score，脑区外置 0（方案 §2.3 第 3 条）。"""
    mask = vol != 0
    out = np.zeros_like(vol, dtype=np.float32)
    if mask.sum() > 0:
        v = vol[mask]
        mu, sd = float(v.mean()), float(v.std())
        sd = sd if sd > 1e-6 else 1.0
        out[mask] = (vol[mask] - mu) / sd
    return out, mask


def preprocess_case(data_root: str, cid: str, cache_dir: str,
                    reorient_to: str = "RAS", overwrite: bool = False,
                    task: str = "binary_wt") -> dict:
    """读一个病例 -> 缓存 npz。返回元信息。

    产物:
      img      [4,D,H,W] float16  四模态 z-score 后；未置零（置零在采样时按场景做）
      seg      [D,H,W]   uint8    binary_wt 为 0/1；四分类为 0/1/2/3（3=ET）
      support  [D,H,W]   bool     四模态非零并集 —— 仅用于训练期 patch 位置采样，绝不作为模型输入
    """
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{cid}.npz")
    meta_path = os.path.join(cache_dir, f"{cid}.json")
    if os.path.exists(out_path) and os.path.exists(meta_path) and not overwrite:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("task", "binary_wt") != task:
            raise ValueError(f"{cid}: 缓存任务不匹配，请使用独立实验目录")
        return meta

    src = os.path.join(data_root, cid)
    vols, affines = [], []
    for m in MODALITIES:
        img = nib.load(os.path.join(src, f"{cid}_{m}.nii.gz"))
        if reorient_to:
            img = nib.as_closest_canonical(img)  # 统一到 RAS，确定性操作
        vols.append(np.asanyarray(img.dataobj).astype(np.float32))
        affines.append(img.affine)

    seg_img = nib.load(os.path.join(src, f"{cid}_seg.nii.gz"))
    if reorient_to:
        seg_img = nib.as_closest_canonical(seg_img)
    seg_raw = np.asanyarray(seg_img.dataobj)

    # 形状 / 仿射一致性检查，不一致直接报错而不是悄悄处理
    shapes = {v.shape for v in vols} | {seg_raw.shape}
    if len(shapes) != 1:
        raise RuntimeError(f"{cid}: 模态与标注形状不一致 {shapes}")
    for a in affines[1:] + [seg_img.affine]:
        if not np.allclose(a, affines[0], atol=1e-3):
            raise RuntimeError(f"{cid}: 仿射矩阵不一致")

    # 标签值检查：出现未预期的取值时报警
    uniq = sorted(int(u) for u in np.unique(seg_raw))
    unexpected = [u for u in uniq if u not in LABEL_MAPPING["raw_values_present"]]
    if unexpected:
        raise RuntimeError(f"{cid}: seg 含未预期标签值 {unexpected}")

    seg = encode_segmentation(seg_raw, task)

    proc, masks = [], []
    for v in vols:
        p, mk = _zscore_brain(v)
        proc.append(p.astype(np.float16))
        masks.append(mk)
    img = np.stack(proc, axis=0)
    support = np.any(np.stack(masks, axis=0), axis=0)

    np.savez_compressed(out_path, img=img, seg=seg, support=support)
    meta = {
        "case_id": cid,
        "task": task,
        "shape": list(img.shape[1:]),
        "spacing": [1.0, 1.0, 1.0],
        "orientation": reorient_to or "original",
        "seg_raw_values": uniq,
        "tumor_voxels": int((seg > 0).sum()),
        "support_voxels": int(support.sum()),
        "path": out_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_case(cache_dir: str, cid: str) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(cache_dir, f"{cid}.npz")) as z:
        return {"img": z["img"].astype(np.float32), "seg": z["seg"], "support": z["support"]}


# ---------------------------------------------------------------- 场景 / patch

def avail_vector(scenario: str) -> np.ndarray:
    return np.asarray(SCENARIOS[scenario]["avail"], dtype=np.float32)


def apply_scenario(img: np.ndarray, scenario: str) -> np.ndarray:
    """归一化之后置零被隐藏模态（方案 §2.3 第 5 条）。"""
    out = img.copy()
    av = SCENARIOS[scenario]["avail"]
    for i, a in enumerate(av):
        if a == 0:
            out[i] = 0.0
    return out


def patch_slices(center: np.ndarray, size: int, shape: Sequence[int]) -> Tuple[slice, ...]:
    """以 center 为中心取 size^3 的 patch，中心被夹紧到合法范围。

    夹紧保证被选中的体素一定落在 patch 内（体积在该轴上不短于 size）。
    """
    half = size // 2
    out = []
    for c, s in zip(center, shape):
        c = int(np.clip(c, half, s - size + half))
        out.append(slice(c - half, c - half + size))
    return tuple(out)


class CaseData:
    """一个病例的全部内存数据。坐标数组在加载时算一次，避免每个样本重算 argwhere。"""

    def __init__(self, cache_dir: str, cid: str, task: str = "binary_wt"):
        with open(os.path.join(cache_dir, f"{cid}.json")) as f:
            meta = json.load(f)
        if meta.get("task", "binary_wt") != task:
            raise ValueError(f"{cid}: 缓存任务与模型不匹配")
        d = load_case(cache_dir, cid)
        self.cid = cid
        self.img = d["img"]              # [4,D,H,W] float32
        self.seg = d["seg"]              # [D,H,W] uint8
        self.support = d["support"]      # [D,H,W] bool，仅用于训练期 patch 位置采样
        self.tumor_xyz = np.argwhere(self.seg > 0).astype(np.int16)
        self.support_xyz = np.argwhere(self.support).astype(np.int16)

    @property
    def shape(self):
        return self.seg.shape


class VolumeStore:
    """把所有需要的病例常驻内存。"""

    def __init__(self, cache_dir: str, cases: Sequence[str], task: str = "binary_wt"):
        self.cases = list(cases)
        self._d: Dict[str, CaseData] = {cid: CaseData(cache_dir, cid, task) for cid in self.cases}

    def __getitem__(self, cid: str) -> CaseData:
        return self._d[cid]


def sample_center(rng: np.random.Generator, tumor_xyz: np.ndarray, support_xyz: np.ndarray,
                  prob_tumor: float) -> np.ndarray:
    if tumor_xyz.shape[0] > 0 and rng.random() < prob_tumor:
        return tumor_xyz[rng.integers(tumor_xyz.shape[0])]
    return support_xyz[rng.integers(support_xyz.shape[0])]


def sample_augmentation(rng: np.random.Generator, n_mod: int):
    """空间翻转 + 小幅强度扰动（方案 §4.3）。空间变换同步作用于标签。

    抽取次序固定（翻转 -> 尺度 -> 平移 -> 噪声开关），保证 A/B 拿到同一份增强。
    """
    flips = tuple(bool(b) for b in (rng.random(3) < 0.5))
    scales = 1.0 + rng.uniform(-0.1, 0.1, size=n_mod).astype(np.float32)
    shifts = rng.normal(0.0, 0.1, size=n_mod).astype(np.float32)
    use_noise = bool(rng.random() < 0.3)
    noise_sd = float(rng.uniform(0.0, 0.1)) if use_noise else 0.0
    return {"flips": flips, "scales": scales, "shifts": shifts, "noise_sd": noise_sd}


def apply_augmentation(patch_img: np.ndarray, patch_seg: np.ndarray, aug: dict,
                       support: np.ndarray | None = None):
    """patch_img [4,64,64,64]；patch_seg [64,64,64] uint8。就地返回新数组。

    support 非 None 时（诊断变体 `aug_keep_background_zero`），仿射与噪声只作用在
    脑支撑内，脑外保持严格 0 —— 缓存里脑外本来就是 0，而增强的 shift 会把它变成
    非零，推理时却又是 0，这个错位正是本变体要排除的因素。
    support 与 patch_seg 同形状，必须跟着 patch_seg 做同样的翻转。
    """
    ax = [1, 2, 3]
    for a, f in zip(ax, aug["flips"]):
        if f:
            patch_img = np.flip(patch_img, axis=a)
            patch_seg = np.flip(patch_seg, axis=0 if a == 1 else (1 if a == 2 else 2))
            if support is not None:
                support = np.flip(support, axis=0 if a == 1 else (1 if a == 2 else 2))
    scales = aug["scales"].reshape(-1, 1, 1, 1)
    shifts = aug["shifts"].reshape(-1, 1, 1, 1)
    patch_img = patch_img * scales + shifts
    if support is not None:
        m3 = support.reshape(1, *support.shape)          # [1,p,p,p]，按通道广播
        patch_img = np.where(m3, patch_img, 0.0)
    if aug["noise_sd"] > 0:
        # 噪声只加在可用模态上；被隐藏模态必须保持严格为 0
        noise = aug["rng"].normal(0.0, aug["noise_sd"], size=patch_img.shape).astype(np.float32)
        noise = noise * aug["avail_mask"]
        if support is not None:
            noise = noise * support.reshape(1, *support.shape)
        patch_img = patch_img + noise
    return np.ascontiguousarray(patch_img, dtype=np.float32), np.ascontiguousarray(patch_seg)


def step_rngs(seed: int, step: int):
    """数据 RNG 与模型噪声 RNG 分开（方案 §4.3）。

    两者都只依赖 (seed, step)，因此方法 A 与方法 B 在相同 seed 下看到的
    患者、patch、增强、缺失场景完全一致；B 额外抽噪声不会改变数据顺序。
    """
    data_rng = np.random.default_rng([seed, step, 1])
    model_rng = np.random.default_rng([seed, step, 2])
    return data_rng, model_rng


def make_batch(store, cases: Sequence[str], seed: int, step: int,
               patch: int, prob_tumor: float, batch_size: int, augment: bool = True,
               n_classes: int = 2, scenario_order=None,
               keep_background_zero: bool = False):
    """构造一个确定性 batch。返回 numpy 数组，A/B 共用同一份。

    cases 是候选病例池，batch_size 是实际样本数（方案 §4.3：有效 batch size = 4）。
    """
    rng, _ = step_rngs(seed, step)
    scenario_order = SCENARIO_ORDER if scenario_order is None else scenario_order
    b = batch_size
    out_img = np.zeros((b, 4, patch, patch, patch), dtype=np.float32)
    out_seg = np.zeros((b, n_classes, patch, patch, patch), dtype=np.float32)
    out_avail = np.zeros((b, 4), dtype=np.float32)
    scenarios: List[str] = []

    for i in range(b):
        cid = cases[rng.integers(len(cases))]
        case = store[cid]
        scen = scenario_order[rng.integers(len(scenario_order))]
        scenarios.append(scen)

        seg = case.seg
        center = sample_center(rng, case.tumor_xyz, case.support_xyz, prob_tumor)

        sl = patch_slices(center, patch, seg.shape)
        # 先取 patch 再置零：避免为每个样本复制整份 240x240x155 体积
        pi = case.img[(slice(None),) + sl].copy()
        ps = seg[sl].astype(np.uint8)
        av = avail_vector(scen)
        pi = pi * av.reshape(-1, 1, 1, 1)   # 归一化之后置零被隐藏模态（方案 §2.3-5）

        if augment:
            sup = None
            if keep_background_zero:
                if not hasattr(case, "support"):
                    raise AttributeError(
                        "keep_background_zero=True 需要病例带 support 掩码；该病例对象没有此属性")
                sup = case.support[sl]
            aug = sample_augmentation(rng, 4)
            aug["rng"] = rng
            aug["avail_mask"] = av.reshape(-1, 1, 1, 1)
            pi, ps = apply_augmentation(pi, ps, aug, support=sup)
            # 增强后再置零，保证被隐藏模态严格为 0
            pi = pi * av.reshape(-1, 1, 1, 1)

        out_img[i] = pi
        if ps.max() >= n_classes:
            raise ValueError("标签超出模型类别范围")
        for label in range(n_classes):
            out_seg[i, label] = (ps == label).astype(np.float32)
        out_avail[i] = av

    return {"mri": out_img, "y1": out_seg, "avail": out_avail, "scenarios": scenarios}


def fm_noise(seed: int, step: int, batch: int, spatial, device, dtype, n_ch: int = 2):
    """B 专用的高斯噪声 y0 与时间 t（方案 §4.2）。与数据 RNG 独立，
    因此 B 额外抽噪声不会改变数据顺序。"""
    import torch
    _, model_rng = step_rngs(seed, step)
    g = torch.Generator(device=device).manual_seed(int(model_rng.integers(2 ** 31 - 1)))
    y0 = torch.randn((batch, n_ch) + tuple(spatial), generator=g,
                     device=device, dtype=torch.float32).to(dtype)
    t = torch.rand((batch, 1, 1, 1, 1), generator=g, device=device, dtype=torch.float32).to(dtype)
    return y0, t


def case_noise_key(cid: str, eval_seed: int) -> int:
    """同一病例、同一 eval_seed 下初始噪声固定；与场景无关（方案 §5.2）。"""
    h = hashlib.sha256(f"{cid}|{eval_seed}".encode()).hexdigest()
    return int(h[:12], 16) % (2 ** 31 - 1)
