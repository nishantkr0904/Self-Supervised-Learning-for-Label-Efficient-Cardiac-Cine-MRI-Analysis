"""CPU unit and smoke tests for src/evaluate_robustness.py."""

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src import evaluate_robustness as er
from src.segmentation_model import SegmentationUNet


def _make_synthetic_test_env(
    root: Path,
    run_ids,
    patient_ids=("patient001", "patient002"),
):
    """Create minimal processed ACDC slices, split file, checkpoints, and config."""
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = root / "checkpoints"
    results_dir = root / "results"
    tables_dir = root / "tables"

    yy, xx = np.ogrid[:32, :32]
    lv = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 16
    myo = (((yy - 16) ** 2 + (xx - 16) ** 2) <= 42) & (~lv)
    rv = ((yy - 16) ** 2 + (xx - 24) ** 2) <= 12
    base_mask = np.zeros((32, 32), dtype=np.int64)
    base_mask[rv] = 3
    base_mask[myo] = 2
    base_mask[lv] = 1

    rng = np.random.default_rng(7)
    for pid in patient_ids:
        for s_idx in range(2):
            img = (base_mask.astype(np.float32) * 0.4 + rng.normal(0.0, 0.25, size=(32, 32))).astype(np.float32)
            np.savez(
                processed_dir / f"{pid}_frame00_slice{s_idx:02d}.npz",
                image=img,
                mask=base_mask,
                patient_id=pid,
                frame_idx=0,
                slice_idx=s_idx,
                has_label=True,
                phase="ED",
            )

    split_path = root / "test_patients.txt"
    split_path.write_text("\n".join(patient_ids) + "\n", encoding="utf-8")

    torch.manual_seed(11)
    for idx, run_id in enumerate(run_ids):
        model = SegmentationUNet(in_channels=1, num_classes=4, encoder_channels=[8, 16, 32, 64])
        # Slightly shift bias per run so outputs stay valid yet distinct.
        with torch.no_grad():
            for param in model.parameters():
                param.add_(idx * 0.005)
        run_ckpt_dir = ckpt_dir / run_id
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_name": run_id,
                "model_state_dict": model.state_dict(),
            },
            run_ckpt_dir / f"{run_id}_best.pth",
        )

    config = {
        "data": {
            "processed_dir": str(processed_dir),
            "test_split": str(split_path),
            "num_classes": 4,
        },
        "model": {
            "architecture": "2d_unet",
            "in_channels": 1,
            "num_classes": 4,
            "encoder_channels": [8, 16, 32, 64],
            "use_ssl_pretrained": False,
        },
        "robustness": {
            "seeds": [42, 123, 456],
            "noise_levels": [0.0, 0.05, 0.10],
            "intensity_scalings": [0.9, 1.0, 1.1],
            "temporal_intervals": [1, 2],
        },
        "logging": {
            "checkpoint_dir": str(ckpt_dir),
            "results_dir": str(results_dir),
            "tables_dir": str(tables_dir),
        },
    }
    cfg_path = root / "experiments.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return cfg_path, split_path, results_dir, tables_dir


class EvaluateRobustnessTests(unittest.TestCase):
    def test_build_perturbation_conditions_from_experiments_yaml(self):
        with open("configs/experiments.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        conditions = er.build_perturbation_conditions(config)
        names = [c["name"] for c in conditions]
        self.assertEqual(
            names,
            [
                "clean",
                "gaussian_noise_0.05",
                "gaussian_noise_0.10",
                "intensity_scale_0.90",
                "intensity_scale_1.10",
            ],
        )

    def test_apply_perturbation_determinism_and_scaling(self):
        x = torch.tensor([[[[-1.5, 0.0], [0.5, 2.0]]]], dtype=torch.float32).repeat(16, 1, 16, 16)

        clean = er.apply_perturbation(x, noise_std=0.0, intensity_scale=1.0)
        self.assertTrue(torch.equal(clean, x))
        self.assertNotEqual(clean.data_ptr(), x.data_ptr())

        scaled_down = er.apply_perturbation(x, noise_std=0.0, intensity_scale=0.9)
        scaled_up = er.apply_perturbation(x, noise_std=0.0, intensity_scale=1.1)
        self.assertTrue(torch.allclose(scaled_down, x * 0.9, atol=1e-6))
        self.assertTrue(torch.allclose(scaled_up, x * 1.1, atol=1e-6))

        g1 = torch.Generator(device="cpu").manual_seed(42)
        g2 = torch.Generator(device="cpu").manual_seed(42)
        g3 = torch.Generator(device="cpu").manual_seed(42)
        n05_a = er.apply_perturbation(x, noise_std=0.05, intensity_scale=1.0, generator=g1)
        n05_b = er.apply_perturbation(x, noise_std=0.05, intensity_scale=1.0, generator=g2)
        n10 = er.apply_perturbation(x, noise_std=0.10, intensity_scale=1.0, generator=g3)

        self.assertTrue(torch.equal(n05_a, n05_b))
        diff_05 = n05_a - x
        diff_10 = n10 - x
        self.assertAlmostEqual(float(diff_05.std()), 0.05, delta=0.005)
        self.assertTrue(torch.allclose(diff_10, 2.0 * diff_05, atol=1e-6))

    def test_main_all_runs_and_write_table_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path, split_path, results_dir, tables_dir = _make_synthetic_test_env(
                root, run_ids=er.DEFAULT_RUN_IDS
            )

            with patch.object(er, "CANONICAL_TEST_SPLIT", split_path):
                rc = er.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--all",
                        "--device",
                        "cpu",
                        "--batch-size",
                        "2",
                        "--seed",
                        "42",
                        "--write-table",
                    ]
                )
            self.assertEqual(rc, 0)

            for run_id in er.DEFAULT_RUN_IDS:
                json_path = results_dir / run_id / "robustness_metrics.json"
                self.assertTrue(json_path.exists(), f"Missing {json_path}")
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["run_id"], run_id)
                self.assertEqual(payload["n_patients"], 2)
                self.assertEqual(payload["n_slices"], 4)
                self.assertEqual(payload["voxel_spacing"], [1.5, 1.5])
                self.assertEqual(
                    list(payload["conditions"].keys()),
                    [
                        "clean",
                        "gaussian_noise_0.05",
                        "gaussian_noise_0.10",
                        "intensity_scale_0.90",
                        "intensity_scale_1.10",
                    ],
                )
                clean_cond = payload["conditions"]["clean"]
                for key in ["Mean_Dice", "LV_Dice", "Myocardium_Dice", "RV_Dice", "Mean_HD95"]:
                    self.assertIn(key, clean_cond["mean"])
                    self.assertIn(key, clean_cond["std"])
                    self.assertEqual(clean_cond["delta_vs_clean"][key], 0.0)

            csv_path = tables_dir / "robustness_table.csv"
            md_path = tables_dir / "robustness_table.md"
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            self.assertIn("supervised_100pct_seed42", {r["Run ID"] for r in rows})
            self.assertIn("full_pipeline_100pct_seed42", {r["Run ID"] for r in rows})

            # Repeat single-run evaluation into an isolated results directory and verify determinism.
            alt_results = root / "alt_results"
            with patch.object(er, "CANONICAL_TEST_SPLIT", split_path):
                rc2 = er.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--run-id",
                        "full_pipeline_100pct_seed42",
                        "--device",
                        "cpu",
                        "--batch-size",
                        "2",
                        "--seed",
                        "42",
                        "--results-dir",
                        str(alt_results),
                    ]
                )
            self.assertEqual(rc2, 0)
            p1 = json.loads((results_dir / "full_pipeline_100pct_seed42" / "robustness_metrics.json").read_text())
            p2 = json.loads((alt_results / "full_pipeline_100pct_seed42" / "robustness_metrics.json").read_text())
            self.assertEqual(p1["conditions"], p2["conditions"])

    def test_fail_closed_provenance_cohort_and_cli_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path, split_path, results_dir, _ = _make_synthetic_test_env(
                root, run_ids=["supervised_100pct_seed42"]
            )
            ckpt_file = root / "checkpoints" / "supervised_100pct_seed42" / "supervised_100pct_seed42_best.pth"
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            ckpt["experiment_name"] = "wrong_run_id"
            torch.save(ckpt, ckpt_file)

            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                with patch.object(er, "CANONICAL_TEST_SPLIT", split_path):
                    er.main(
                        [
                            "--config",
                            str(cfg_path),
                            "--run-id",
                            "supervised_100pct_seed42",
                            "--device",
                            "cpu",
                        ]
                    )

            mismatched_split = root / "canonical_extra.txt"
            mismatched_split.write_text("patient001\npatient002\npatient999\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical split"):
                with patch.object(er, "CANONICAL_TEST_SPLIT", mismatched_split):
                    er.main(
                        [
                            "--config",
                            str(cfg_path),
                            "--run-id",
                            "supervised_100pct_seed42",
                            "--device",
                            "cpu",
                        ]
                    )

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                er.parse_args(["--all", "--checkpoint-path", "some.pth"])
            with self.assertRaises(SystemExit):
                er.parse_args([])

    def test_cli_help_entrypoint(self):
        proc = subprocess.run(
            [sys.executable, "src/evaluate_robustness.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--write-table", proc.stdout)
        self.assertIn("--processed-dir", proc.stdout)


if __name__ == "__main__":
    unittest.main()
