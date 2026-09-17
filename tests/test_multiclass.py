import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from fmexp.config import Config, save_config
from fmexp.data import (encode_segmentation, decode_segmentation, make_batch,
                        avail_vector, preprocess_case, VolumeStore)
from fmexp.infer import init_noise_for_case, predict_volume_A, predict_volume_B
from fmexp.metrics import region_masks, segmentation_metrics
from fmexp.train import forward_loss_A, forward_loss_B, validation_assignments, validate, train_one
from fmexp.unet import build_model
from run_eval import build_summary, paired_diffs, rebuild_combined, write_csv, METRIC_COLS, evaluate_split
from fmexp.multiclass_report import write_report


class MulticlassTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = Config(task="multiclass_missing_one", patch=8, sw_window=8,
                          base_channels=2, norm_groups=2)

    def test_raw_labels_regions_and_empty_region_policy(self):
        raw = np.array([0, 1, 2, 4], np.uint8).reshape(1, 1, 4)
        seg = encode_segmentation(raw, self.cfg.task)
        np.testing.assert_array_equal(decode_segmentation(seg, 4), raw)
        masks = region_masks(seg)
        self.assertEqual([int(masks[r].sum()) for r in ("WT", "TC", "ET")], [3, 2, 1])
        for m in segmentation_metrics(seg, seg, 4):
            self.assertEqual(m["dice"], 1)
            self.assertEqual(m["hd95"], 0)
        empty = np.zeros_like(seg)
        for m in segmentation_metrics(empty, empty, 4):
            self.assertEqual(m["dice"], 1)
            self.assertTrue(m["both_empty"])
            self.assertFalse(m["hd95_failed"])
        for pred, gt in [(empty, seg), (seg, empty)]:
            for m in segmentation_metrics(pred, gt, 4):
                self.assertEqual(m["dice"], 0)
                self.assertTrue(m["hd95_failed"])
                self.assertTrue(np.isnan(m["hd95"]))

    def test_sampling_onehot_and_both_training_losses(self):
        seg = np.indices((8, 8, 8))[0].astype(np.uint8) % 4
        case = SimpleNamespace(img=np.ones((4, 8, 8, 8), np.float32), seg=seg,
                               tumor_xyz=np.argwhere(seg > 0), support_xyz=np.argwhere(seg >= 0))
        kwargs = dict(store={"c": case}, cases=["c"], seed=0, step=3, patch=8,
                      prob_tumor=0.5, batch_size=32, n_classes=4, scenario_order=self.cfg.scenarios)
        batch = make_batch(**kwargs)
        repeated = make_batch(**kwargs)
        for k in ("mri", "y1", "avail"):
            np.testing.assert_array_equal(batch[k], repeated[k])
        self.assertEqual(set(batch["scenarios"]), set(self.cfg.scenarios))
        self.assertTrue(np.all(batch["avail"].sum(1) == 3))
        self.assertTrue(np.all(batch["y1"].sum(1) == 1))
        for image, av in zip(batch["mri"], batch["avail"]):
            self.assertTrue(np.all(image[av == 0] == 0))
        mri, y1, avail = (torch.from_numpy(batch[k][:2]) for k in ("mri", "y1", "avail"))
        for method in ("A", "B"):
            model = build_model(method, self.cfg)
            if method == "A":
                loss, _ = forward_loss_A(model, mri, avail, y1, self.cfg)
            else:
                loss, _ = forward_loss_B(model, mri, avail, y1, 0, 3, self.cfg, "cpu")
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        allocation = validation_assignments([str(i) for i in range(20)], 42, self.cfg.scenarios)
        self.assertEqual(sorted(Counter(allocation.values()).values()), [5, 5, 5, 5])

    def test_four_channel_sliding_window_and_euler_oracle(self):
        shape = (12, 12, 12)
        target = torch.tensor(np.indices(shape)[0] % 4)
        onehot = torch.nn.functional.one_hot(target, 4).permute(3, 0, 1, 2).float()
        noise = init_noise_for_case("c", shape, 1234, "cpu", 4)
        av = torch.ones(4)

        class Oracle(torch.nn.Module):
            def forward(self, x):
                return x[:, :4]

        pred = predict_volume_B(Oracle(), onehot - noise, av, self.cfg, noise, chunk=2)
        np.testing.assert_array_equal(pred, target.numpy())
        pred_a = predict_volume_A(Oracle(), onehot * 10, av, self.cfg, chunk=2)
        np.testing.assert_array_equal(pred_a, target.numpy())

    def test_preprocessing_does_not_reuse_binary_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "c"
            source.mkdir()
            raw = (np.indices((8, 8, 8))[0] % 4).astype(np.uint8)
            raw[raw == 3] = 4
            for m in ("t1", "t2", "t1ce", "flair", "seg"):
                data = raw if m == "seg" else np.ones(raw.shape, np.float32)
                nib.save(nib.Nifti1Image(data, np.eye(4)), source / f"c_{m}.nii.gz")
            cache = str(root / "cache")
            meta = preprocess_case(tmp, "c", cache, task=self.cfg.task)
            self.assertEqual(meta["tumor_voxels"], int((raw > 0).sum()))
            store = VolumeStore(cache, ["c"], self.cfg.task)
            np.testing.assert_array_equal(decode_segmentation(store["c"].seg, 4), raw)
            with self.assertRaises(ValueError):
                VolumeStore(cache, ["c"])
            with self.assertRaises(ValueError):
                preprocess_case(tmp, "c", cache)

    def test_region_summary_pairing_and_report(self):
        rows = []
        gt = np.array([0, 1, 2, 3], np.uint8).reshape(1, 1, 4)
        for model in ("A", "B"):
            for scenario in self.cfg.scenarios:
                for m in segmentation_metrics(gt, gt, 4):
                    rows.append(dict(m, patient_id="case", model=model, seed=0, scenario=scenario,
                                     split="test", ckpt="best.pt", seconds=1., peak_gpu_gb=1.))
        self.assertEqual(len(build_summary(rows)), 24)
        self.assertEqual(len(paired_diffs(rows)), 12)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_config(self.cfg, str(root / "config_preprocess.yaml"))
            (root / "splits.json").write_text(json.dumps({"test": ["case"]}))
            dest = root / "evaluation" / "test" / "main"
            cols = ["patient_id", "model", "seed", "scenario", "region", "split", "ckpt"] + METRIC_COLS + ["seconds", "peak_gpu_gb"]
            write_csv(rows, str(dest / "metrics_per_case_seed0_A_B.csv"), cols)
            # Identical repetitions must not inflate patient counts.
            write_csv(rows, str(dest / "metrics_per_case_seed0_duplicate.csv"), cols)
            combined, summary, boot = rebuild_combined(str(dest))
            self.assertEqual(len(combined), 24)
            self.assertTrue(all(r["n"] == 1 for r in summary))
            report = write_report(root).read_text()
            self.assertIn("missing_flair", report)
            self.assertIn("ET", report)
            self.assertIn("HD95", report)

    def test_training_checkpoint_to_full_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(task="multiclass_missing_one", data_root=str(root / "raw"),
                         patch=8, sw_window=8, base_channels=2, norm_groups=2,
                         max_steps=1, batch_size=2, fm_steps=2, device="cpu")
            splits = {"train": ["train"], "val": [f"val{i}" for i in range(4)], "test": ["test"]}
            cache = root / "cache"
            cache.mkdir()
            seg = (np.indices((8, 8, 8))[0] % 4).astype(np.uint8)
            for cid in sum(splits.values(), []):
                np.savez(cache / f"{cid}.npz", img=np.ones((4, 8, 8, 8), np.float32),
                         seg=seg, support=np.ones(seg.shape, bool))
                (cache / f"{cid}.json").write_text(json.dumps({"task": cfg.task}))
            source = root / "raw" / "test"
            source.mkdir(parents=True)
            nib.save(nib.Nifti1Image(decode_segmentation(seg, 4), np.eye(4)), source / "test_seg.nii.gz")
            for method in ("A", "B"):
                run_dir = root / "runs" / f"{method}_seed0"
                save_config(cfg, str(run_dir / "config.yaml"))
                result = train_one(method, cfg, 0, splits, str(cache), str(run_dir),
                                   torch.device("cpu"), log=lambda _: None)
                self.assertEqual(result["best_step"], 1)
                full = json.loads((run_dir / "best_full_validation.json").read_text())
                self.assertEqual(len(full["rows"]), 48)
                rows = evaluate_split(method, 0, "best.pt", cfg, splits, "test", ["test"],
                                      str(cache), str(root), torch.device("cpu"))
                self.assertEqual(len(rows), 12)
                files = list((root / "predictions" / "test" / "test").glob(f"{method}_*.nii.gz"))
                self.assertEqual(len(files), 4)
                for file in files:
                    pred = np.asanyarray(nib.load(file).dataobj)
                    self.assertTrue(np.isin(pred, [0, 1, 2, 4]).all())


if __name__ == "__main__":
    unittest.main()
