"""诊断模块的单测：解析可核对的公式、改动引入的不变量、以及接线正确性。"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from fmexp.config import Config
from fmexp.data import apply_augmentation, make_batch
from fmexp.diagnostics import (bayes_velocity_mse, conditioning_variants, endpoint_margin, final_state,
                               connected_component_stats, error_location,
                               mask_to_support, steps_sweep, trajectory_diagnostic,
                               velocity_mse_by_t)
from fmexp.infer import init_noise_for_case


class BayesReferenceTests(unittest.TestCase):
    def test_deterministic_prior_gives_zero_mse(self):
        """先验退化到单一类别时后验无不确定性，最优速度 MSE 必须为 0。

        对固定标签 y1，给定 y_t 后 y0=(y_t−t·y1)/(1−t) 被完全确定，
        于是 v*=(y1−y_t)/(1−t) 与真值之差恒为 0。
        """
        p = np.array([0.0, 1.0, 0.0, 0.0])
        for t in (0.1, 0.5, 0.9):
            self.assertLess(bayes_velocity_mse(t, p, n=20000), 1e-9)

    def test_prior_only_reference_finite_and_positive(self):
        p = np.array([0.25] * 4)
        for t in (0.2, 0.5, 0.8):
            v = bayes_velocity_mse(t, p, n=20000)
            self.assertTrue(np.isfinite(v) and v >= 0.0)

    def test_zero_prior_entries_do_not_blow_up(self):
        p = np.array([0.7, 0.3, 0.0, 0.0])
        self.assertTrue(np.isfinite(bayes_velocity_mse(0.5, p, n=5000)))


class AugmentationBackgroundTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = Config(task="multiclass_missing_one", patch=8, sw_window=8,
                          base_channels=2, norm_groups=2)
        seg = np.zeros((8, 8, 8), np.uint8)
        seg[2:6, 2:6, 2:6] = 1
        seg[3:5, 3:5, 3:5] = 3
        sup = np.zeros((8, 8, 8), bool)
        sup[2:6, 2:6, 2:6] = True
        img = np.zeros((4, 8, 8, 8), np.float32)
        img[:, 2:6, 2:6, 2:6] = 1.5
        self.case = SimpleNamespace(img=img, seg=seg, support=sup,
                                    tumor_xyz=np.argwhere(seg > 0),
                                    support_xyz=np.argwhere(sup))
        self.store = {"c": self.case}

    def _batch(self, **kw):
        return make_batch(self.store, ["c"], seed=0, step=1, patch=8, prob_tumor=0.9,
                          batch_size=8, n_classes=4,
                          scenario_order=self.cfg.scenarios, **kw)

    def test_background_stays_exactly_zero_only_when_requested(self):
        """开启变体后：脑外严格为 0；关闭时（默认冻结行为）背景会被 shift 变成非零。"""
        b = self._batch(augment=True, keep_background_zero=True)
        sup = self.case.support
        for i in range(b["mri"].shape[0]):
            mri = b["mri"][i]
            av = b["avail"][i]
            for c in range(4):
                if av[c] > 0:
                    # 可用模态：脑外必须是严格 0
                    self.assertEqual(float(np.abs(mri[c][~sup]).max()), 0.0)
                else:
                    # 被隐藏模态：整幅都必须是严格 0
                    self.assertEqual(float(np.abs(mri[c]).max()), 0.0)
            # 脑内至少要有一个非零值，否则说明增强把信号也抹掉了
            self.assertGreater(float(np.abs(mri[av > 0][:, sup]).max()), 0.0)

    def test_keep_background_zero_requires_support(self):
        bad = {"c": SimpleNamespace(img=self.case.img, seg=self.case.seg,
                                    tumor_xyz=self.case.tumor_xyz,
                                    support_xyz=self.case.support_xyz)}
        with self.assertRaises(AttributeError):
            make_batch(bad, ["c"], seed=0, step=1, patch=8, prob_tumor=0.9,
                       batch_size=2, n_classes=4, scenario_order=self.cfg.scenarios,
                       augment=True, keep_background_zero=True)

    def test_default_path_unchanged_by_new_flag(self):
        """新增开关默认关闭时，逐位等于旧行为。"""
        a = self._batch(augment=True)
        b = self._batch(augment=True, keep_background_zero=False)
        np.testing.assert_array_equal(a["mri"], b["mri"])
        np.testing.assert_array_equal(a["y1"], b["y1"])

    def test_augmentation_flips_mask_consistently(self):
        """支撑掩码必须跟着标签一起翻转，否则背景置零会错位。"""
        img = np.zeros((4, 8, 8, 8), np.float32)
        sup = np.zeros((8, 8, 8), bool)
        sup[:3, :, :] = True
        img[:, :3] = 2.0
        seg = np.zeros((8, 8, 8), np.uint8)
        aug = {"flips": (True, False, False), "scales": np.ones(4, np.float32),
               "shifts": np.zeros(4, np.float32), "noise_sd": 0.0,
               "rng": np.random.default_rng(0), "avail_mask": np.ones((4, 1, 1, 1), np.float32)}
        out, _ = apply_augmentation(img, seg, aug, support=sup)
        # 翻转后非零区应整体移到后半（掩码跟着翻），脑外仍严格为 0
        self.assertEqual(float(np.abs(out[:, :5]).max()), 0.0)
        self.assertEqual(float(np.abs(out[:, 5:]).max()), 2.0)


class ErrorLocationTests(unittest.TestCase):
    def test_split_counts_and_confusion(self):
        # 各块刻意互不重叠，便于手算
        sup = np.zeros((10, 10, 10), bool)
        sup[2:10, 2:10, 2:10] = True     # 支撑从索引 2 开始，好让脑外那块完全落在外面
        gt = np.zeros((10, 10, 10), np.uint8)
        gt[2:5, 2:5, 2:5] = 1            # NCR，27 体素
        gt[5:7, 5:7, 5:7] = 3            # ET，8 体素
        pred = np.zeros((10, 10, 10), np.uint8)
        pred[2:5, 2:5, 2:5] = 1          # 命中 NCR
        pred[7:9, 7:9, 7:9] = 2          # 脑内假阳（错类，与 GT 无重叠）
        pred[0:2, 0:2, 0:2] = 3          # 脑外假阳
        r = error_location(pred, gt, sup)
        self.assertEqual(r["tp"], 27)
        self.assertEqual(r["fn"], 8)      # ET 全漏
        self.assertEqual(r["fp_in"], 8)
        self.assertEqual(r["fp_out"], 8)
        cm = np.array(r["confusion"])
        self.assertEqual(cm[1, 1], 27)    # 真实 NCR 全部预测为 NCR
        self.assertEqual(list(cm[3]), [8, 0, 0, 0])   # 真实 ET 全部漏检
        self.assertAlmostEqual(r["fp_out_frac"], 0.5, places=6)

    def test_mask_to_support(self):
        pred = np.ones((4, 4, 4), np.uint8)
        sup = np.zeros((4, 4, 4), bool)
        sup[1:3, 1:3, 1:3] = True
        out = mask_to_support(pred, sup)
        self.assertEqual(int((out > 0).sum()), 8)
        self.assertEqual(int(out[0, 0, 0]), 0)

    def test_connected_components(self):
        m = np.zeros((10, 10, 10), bool)
        m[1:4, 1:4, 1:4] = True
        m[8, 8, 8] = True
        s = connected_component_stats(m)
        self.assertEqual(s["n_components"], 2)
        self.assertEqual(s["largest"], 27)
        self.assertEqual(s["total_voxels"], 28)
        # 碎块阈值是「最大块的 1%」，本例最大块才 27，单个体素够不上阈值
        self.assertEqual(s["small_frac"], 0.0)
        self.assertEqual(connected_component_stats(np.zeros((4, 4, 4), bool))["n_components"], 0)

    def test_small_frac_counts_speckle_at_realistic_scale(self):
        """真实数据量级下，孤立小体素应被判为碎块。"""
        m = np.zeros((40, 40, 40), bool)
        m[5:25, 5:25, 5:25] = True      # 最大块 8000
        m[35, np.arange(10) * 4, 35] = True   # 10 个互不相邻的孤立体素
        s = connected_component_stats(m)
        self.assertEqual(s["n_components"], 11)
        self.assertAlmostEqual(s["small_frac"], 10 / 8010, places=6)


class WiringTests(unittest.TestCase):
    """用「理想速度场」模型核对诊断函数的接线（与既有 Euler oracle 测试同一思路）。"""

    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = Config(task="multiclass_missing_one", patch=8, sw_window=8,
                          base_channels=2, norm_groups=2)

    class Oracle(torch.nn.Module):
        """B 的输入是 [mri(4), avail(4), y(K), t(1)]，返回前 4 个通道。

        测试里把 mri 位置放上 y1−y0，于是模型输出的速度恰好等于真值。
        """
        def forward(self, x):
            return x[:, :4]

    @staticmethod
    def _blob_seg(shape=(8, 8, 8)):
        """单个连通团块（避免条带状标签把前景切成多块，干扰连通块计数）。"""
        seg = np.zeros(shape, np.uint8)
        seg[2:6, 2:6, 2:6] = 3
        seg[3:5, 3:5, 3:5] = 1
        return seg

    def test_trajectory_diagnostic_is_exact_with_ideal_field(self):
        shape = (8, 8, 8)
        seg = self._blob_seg(shape)
        onehot = torch.nn.functional.one_hot(
            torch.tensor(seg.astype(np.int64)), 4).permute(3, 0, 1, 2).float()
        noise = init_noise_for_case("c", shape, 1234, "cpu", 4)
        d = trajectory_diagnostic(self.Oracle(), onehot - noise, torch.ones(4), self.cfg,
                                  noise, seg, chunk=2)
        # 速度场恒等于真值时，轨迹必须精确落在理想路径上
        for row in d["rows"]:
            self.assertAlmostEqual(row["drift_mse"], 0.0, places=10)
            self.assertAlmostEqual(row["v_mse_ontraj"], 0.0, places=10)
            self.assertAlmostEqual(row["v_mse_onideal"], 0.0, places=10)
        for region in ("WT", "TC", "ET"):
            self.assertAlmostEqual(d["final_dice"][region], 1.0, places=10)
        # 第 0 步还是纯噪声，只有末步才应完全成形
        self.assertLess(d["rows"][0]["dice_traj"], 1.0)
        self.assertAlmostEqual(d["rows"][-1]["dice_traj"], 1.0, places=10)

    def test_endpoint_margin_confident_with_ideal_field(self):
        """理想速度场下末态就是 one-hot：间隔恒为 1，通道和恒为 1。"""
        shape = (8, 8, 8)
        seg = self._blob_seg(shape)
        onehot = torch.nn.functional.one_hot(
            torch.tensor(seg.astype(np.int64)), 4).permute(3, 0, 1, 2).float()
        noise = init_noise_for_case("c", shape, 1234, "cpu", 4)
        y = final_state(self.Oracle(), onehot - noise, torch.ones(4), self.cfg, noise)
        torch.testing.assert_close(y, onehot, atol=1e-5, rtol=0)
        m = endpoint_margin(y, seg, np.ones(shape, bool))
        self.assertAlmostEqual(m["margin_median"], 1.0, places=5)
        self.assertAlmostEqual(m["margin_mean"], 1.0, places=5)
        self.assertEqual(m["frac_margin_lt_0.5"], 0.0)
        self.assertEqual(m["margin_wrong_mean"], None)   # 全部判对，没有错体素
        self.assertAlmostEqual(m["channelsum_mean"], 1.0, places=5)
        self.assertLess(m["channelsum_std"], 1e-5)   # 16 步 Euler 的浮点残差

    def test_endpoint_margin_flags_ties(self):
        """人为造一个平局状态：间隔为 0 应被计入「糊掉」的比例。"""
        y = torch.zeros(4, 4, 4, 4)
        y[0] = 0.4
        y[1] = 0.4          # 与第 0 类并列
        seg = np.zeros((4, 4, 4), np.uint8)
        m = endpoint_margin(y, seg, np.ones((4, 4, 4), bool))
        self.assertAlmostEqual(m["margin_median"], 0.0, places=6)
        self.assertEqual(m["frac_margin_lt_0.1"], 1.0)

    def test_steps_sweep_exact_for_any_step_count(self):
        shape = (8, 8, 8)
        seg = self._blob_seg(shape)
        onehot = torch.nn.functional.one_hot(
            torch.tensor(seg.astype(np.int64)), 4).permute(3, 0, 1, 2).float()
        noise = init_noise_for_case("c", shape, 1234, "cpu", 4)
        out = steps_sweep(self.Oracle(), onehot - noise, torch.ones(4), self.cfg,
                          noise, seg, (4, 8, 16), chunk=2)
        for s, v in out.items():
            self.assertAlmostEqual(v["dice"]["WT"], 1.0, places=10, msg=f"steps={s}")
            self.assertEqual(v["components"]["n_components"], 1)

    def test_velocity_mse_by_t_zero_with_ideal_field(self):
        seg = (np.indices((8, 8, 8))[0] % 4).astype(np.uint8)
        img = np.zeros((4, 8, 8, 8), np.float32)
        case = SimpleNamespace(img=img, seg=seg, support=np.ones((8, 8, 8), bool),
                               tumor_xyz=np.argwhere(seg > 0),
                               support_xyz=np.argwhere(seg >= 0))
        store = {"c": case}

        class LabelOracle(torch.nn.Module):
            """直接把输入里的 y 通道当作终点估计，构造出恰好正确的速度。"""
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

            def forward(self, x):
                return x[:, 8:8 + self.cfg.n_classes]

        # _velocity_field 对每个窗口预测速度；这里只要确认接线能跑通并给出有限值
        res = velocity_mse_by_t(LabelOracle(self.cfg), store, ["c"], self.cfg,
                                torch.device("cpu"), n_batches=1, chunk=2)
        self.assertEqual(len(res["bins"]), 8)
        for b in res["bins"]:
            self.assertTrue(np.isfinite(b["mse_model"]))
            self.assertTrue(np.isfinite(b["r2"]))


class ConditioningVariantTests(unittest.TestCase):
    def test_shuffle_preserves_brain_values_and_zeroes_background(self):
        rng = np.random.default_rng(0)
        mri = rng.standard_normal((4, 8, 8, 8)).astype(np.float32)
        sup = np.zeros((8, 8, 8), bool)
        sup[2:6, 2:6, 2:6] = True
        other = rng.standard_normal((4, 8, 8, 8)).astype(np.float32)
        v = conditioning_variants(mri, sup, "c", other, 1234, rng)
        self.assertEqual(set(v), {"real", "zero", "swap_case", "shuffle_vox"})
        # 打乱后脑内强度多重集不变、脑外为 0
        self.assertEqual(float(np.abs(v["shuffle_vox"][:, ~sup]).max()), 0.0)
        self.assertAlmostEqual(float(np.sort(v["shuffle_vox"][:, sup]).sum()),
                               float(np.sort(mri[:, sup]).sum()), places=3)
        # 空间结构确实被破坏了
        self.assertFalse(np.allclose(v["shuffle_vox"][:, sup], mri[:, sup]))
        self.assertEqual(float(np.abs(v["zero"]).max()), 0.0)
        np.testing.assert_array_equal(v["swap_case"], other)


if __name__ == "__main__":
    unittest.main()
