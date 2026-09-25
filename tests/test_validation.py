"""CPU checks for validation coverage and checkpoint selection; no MRI data required."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


import os
import sys

# 让 `python3 -m unittest discover -s tests` 在仓库根目录下也能
# 解析 fm_missing_mri_test.fmexp（父目录需在 sys.path 上）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from fm_missing_mri_test.fmexp.config import Config, SCENARIO_ORDER
from fm_missing_mri_test.fmexp.train import train_one, validate, validation_assignments


class ValidationTests(unittest.TestCase):
    def test_assignment_is_balanced_and_independent_of_input_order(self):
        cases = [f"case{i:02d}" for i in range(20)]
        assignments = validation_assignments(cases, 42)
        self.assertEqual(assignments, validation_assignments(cases[::-1], 42))
        self.assertEqual(sorted(Counter(assignments.values()).values()), [6, 7, 7])
        with self.assertRaises(ValueError):
            validation_assignments(["same"] * 20, 42)

    def test_validation_coverage_for_both_models(self):
        cases = [f"case{i}" for i in range(6)]
        assignments = validation_assignments(cases, 42)
        shape = (2, 2, 2)
        store = {c: SimpleNamespace(img=np.ones((4,) + shape, np.float32),
                                    seg=np.ones(shape, np.uint8)) for c in cases}
        model = torch.nn.Identity().eval()

        def predict_a(model, mri, av, cfg):
            self.assertTrue(torch.all(mri[av == 0] == 0))
            return np.ones(shape)

        with patch("fm_missing_mri_test.fmexp.train.predict_volume_A", side_effect=predict_a), \
             patch("fm_missing_mri_test.fmexp.train.predict_volume_B", return_value=np.ones(shape, np.uint8)):
            for method in ("A", "B"):
                balanced = validate(model, method, Config(), store, cases, "cpu", assignments)
                full = validate(model, method, Config(), store, cases, "cpu")
                self.assertEqual(len(balanced), 6)
                self.assertEqual(len(full), 18)
                self.assertEqual({(r["case_id"], r["scenario"]) for r in balanced},
                                 set(assignments.items()))
                self.assertEqual({(r["case_id"], r["scenario"]) for r in full},
                                 {(c, s) for c in cases for s in SCENARIO_ORDER})
                self.assertFalse(model.training)

    def test_schedule_and_full_validation_do_not_reselect_checkpoint(self):
        cfg = Config(max_steps=5, val_every=2, device="cpu")
        splits = {"train": ["train"], "val": [f"case{i}" for i in range(6)]}
        model = torch.nn.Linear(1, 1)
        calls = []
        # Smoke would win if mistakenly included; last full score must not overwrite best.
        scores = iter([1.0, 0.5, 0.8, 0.6, 0.1])

        def fake_validate(model, method, cfg, store, cases, device, assignments=None):
            calls.append((list(cases), assignments))
            score = next(scores)
            return [{"case_id": c, "scenario": s, "dice": score}
                    for c in cases
                    for s in ([assignments[c]] if assignments is not None else SCENARIO_ORDER)]

        def fake_loss(model, **kwargs):
            return model(torch.ones(1, 1)).square().mean(), {}

        with tempfile.TemporaryDirectory() as tmp, \
             patch("fm_missing_mri_test.fmexp.train.VolumeStore", return_value={}), \
             patch("fm_missing_mri_test.fmexp.train.build_model", return_value=model), \
             patch("fm_missing_mri_test.fmexp.train.make_batch", return_value={}), \
             patch("fm_missing_mri_test.fmexp.train._to_device", return_value={"mri": None, "avail": None, "y1": None}), \
             patch("fm_missing_mri_test.fmexp.train.forward_loss_A", side_effect=lambda m, *args: fake_loss(m)), \
             patch("fm_missing_mri_test.fmexp.train.validate", side_effect=fake_validate):
            result = train_one("A", cfg, 0, splits, "unused", tmp,
                               torch.device("cpu"), log=lambda msg: None)
            logs = [json.loads(line) for line in (Path(tmp) / "log.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in logs if r["type"] == "val"], [2, 4, 5])
            self.assertEqual(result["best_step"], 4)
            self.assertAlmostEqual(result["best_score"], 0.8)
            self.assertAlmostEqual(result["best_full_val_score"], 0.1)
            self.assertEqual(len(calls[0][0]), 3)
            self.assertIsNone(calls[-1][1])
            self.assertTrue(all(c[1] == calls[1][1] for c in calls[1:-1]))
            best = torch.load(Path(tmp) / "best.pt", weights_only=False)
            self.assertEqual(best["step"], 4)
            self.assertAlmostEqual(best["score"], 0.8)
            self.assertEqual(torch.load(Path(tmp) / "final.pt", weights_only=False)["step"], 5)
            self.assertTrue((Path(tmp) / "best_full_validation.json").exists())


if __name__ == "__main__":
    unittest.main()
