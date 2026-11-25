"""CPU unit tests for src/statistical_analysis.py."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import statistical_analysis as sa


def _make_synthetic_experiments(root: Path, n_patients: int = 20):
    exp_dir = root / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = [f"patient{i:03d}" for i in range(1, n_patients + 1)]
    split_path = root / "test_patients.txt"
    split_path.write_text("\n".join(patient_ids) + "\n", encoding="utf-8")

    base_dice = np.linspace(0.60, 0.85, n_patients)
    base_hd = np.linspace(30.0, 8.0, n_patients)

    def write_test_run(run_id: str, dice_offset: np.ndarray, hd_offset: np.ndarray):
        per_patient = {}
        d_vals = base_dice + dice_offset
        h_vals = base_hd + hd_offset
        for pid, d, h in zip(patient_ids, d_vals, h_vals):
            per_patient[pid] = {
                "Mean_Dice": float(d),
                "Mean_HD95": float(h),
            }
        run_dir = exp_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "test_metrics.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "n_patients": n_patients,
                    "per_patient": per_patient,
                    "mean": {"Mean_Dice": float(d_vals.mean()), "Mean_HD95": float(h_vals.mean())},
                    "std": {"Mean_Dice": float(d_vals.std()), "Mean_HD95": float(h_vals.std())},
                }
            ),
            encoding="utf-8",
        )

    ramp = np.linspace(0.06, 0.12, n_patients)
    tiny = np.array([(-1) ** i * 0.002 for i in range(n_patients)])

    write_test_run("supervised_10pct_seed42", np.zeros(n_patients) - 0.10, np.zeros(n_patients) + 12.0)
    write_test_run("full_pipeline_10pct_seed42", np.zeros(n_patients) - 0.10 + ramp, np.zeros(n_patients) + 4.0)
    write_test_run("supervised_25pct_seed42", np.zeros(n_patients) - 0.05, np.zeros(n_patients) + 6.0)
    write_test_run("full_pipeline_25pct_seed42", np.zeros(n_patients) - 0.05 + ramp * 0.7, np.zeros(n_patients) + 2.0)
    write_test_run("supervised_50pct_seed42", np.zeros(n_patients) - 0.02, np.zeros(n_patients) + 3.0)
    write_test_run("full_pipeline_50pct_seed42", np.zeros(n_patients) - 0.02 + ramp * 0.3, np.zeros(n_patients) + 1.0)

    write_test_run("ssl_finetune_100pct_seed42", np.zeros(n_patients), np.zeros(n_patients))
    write_test_run("ssl_motion_100pct_seed42", np.zeros(n_patients) - 0.015, np.zeros(n_patients) + 1.5)
    write_test_run("full_pipeline_raw_100pct_seed42", np.zeros(n_patients) - 0.003, np.zeros(n_patients) + 0.5)
    write_test_run("full_pipeline_100pct_seed42", tiny, np.zeros(n_patients) + 2.0)
    write_test_run("ssl_pseudo_100pct_seed42", tiny + 0.003, np.zeros(n_patients) - 1.5)

    for run_id, shift in [("full_pipeline_100pct_seed42", 0.0), ("supervised_100pct_seed42", -0.025)]:
        conds = {}
        for cond, d_delta, h_delta in [
            ("clean", 0.0, 0.0),
            ("gaussian_noise_0.05", -0.01, 2.0),
            ("gaussian_noise_0.10", -0.02, 3.0),
            ("intensity_scale_0.90", 0.001, -0.5),
            ("intensity_scale_1.10", -0.002, 0.5),
        ]:
            per_patient = {}
            for pid, d, h in zip(patient_ids, base_dice + shift + d_delta, base_hd + h_delta - shift * 100.0):
                per_patient[pid] = {"Mean_Dice": float(d), "Mean_HD95": float(h)}
            conds[cond] = {"per_patient": per_patient}
        run_dir = exp_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "robustness_metrics.json").write_text(
            json.dumps({"run_id": run_id, "conditions": conds}),
            encoding="utf-8",
        )

    return exp_dir, split_path


class StatisticalAnalysisTests(unittest.TestCase):
    def test_holm_bonferroni_step_down(self):
        adj = sa.holm_bonferroni([0.01, 0.04, 0.03, 0.20])
        self.assertAlmostEqual(adj[0], 0.04, places=7)
        self.assertAlmostEqual(adj[1], 0.09, places=7)
        self.assertAlmostEqual(adj[2], 0.09, places=7)
        self.assertAlmostEqual(adj[3], 0.20, places=7)
        self.assertEqual(sa.holm_bonferroni([]), [])

    def test_compute_paired_wilcoxon_and_zero_guard(self):
        a = np.linspace(0.70, 0.89, 20)
        b = np.linspace(0.60, 0.79, 20)
        res = sa.compute_paired_wilcoxon(a, b)
        self.assertEqual(res["n_patients"], 20)
        self.assertAlmostEqual(res["paired_delta"], 0.10, places=6)
        self.assertEqual(res["w_stat"], 0.0)
        self.assertLess(res["raw_p"], 0.001)

        identical = sa.compute_paired_wilcoxon(a, a)
        self.assertEqual(identical["w_stat"], 0.0)
        self.assertEqual(identical["raw_p"], 1.0)

    def test_extract_matched_vectors_rejects_mismatch(self):
        pa = {"p1": {"Mean_Dice": 0.8}, "p2": {"Mean_Dice": 0.7}}
        pb = {"p1": {"Mean_Dice": 0.7}}
        with self.assertRaises(ValueError):
            sa.extract_matched_vectors(pa, pb, metric="Mean_Dice")
        with self.assertRaises(ValueError):
            sa.extract_matched_vectors(pa, pa, metric="Mean_Dice", expected_patients={"p1", "p2", "p3"})

    def test_run_statistical_analysis_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir, split_path = _make_synthetic_experiments(root, n_patients=20)
            out_dir = root / "tables"

            payload = sa.run_statistical_analysis(
                experiments_dir=exp_dir,
                output_dir=out_dir,
                canonical_split=split_path,
                alpha=0.05,
            )
            # (3 + 4 + 4 + 5) comparisons x 2 metrics = 32 tests
            self.assertEqual(len(payload["comparisons"]), 32)
            self.assertEqual(payload["n_patients"], 20)

            for fname in ["statistical_tests.json", "statistical_tests.csv", "statistical_tests.md"]:
                self.assertTrue((out_dir / fname).exists(), f"Missing {fname}")

            with (out_dir / "statistical_tests.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            self.assertEqual(
                list(rows[0].keys()),
                [
                    "Family",
                    "Comparison",
                    "Metric",
                    "Mean A ± Std A",
                    "Mean B ± Std B",
                    "Paired Δ",
                    "W-stat",
                    "Raw p",
                    "Holm p",
                    "Significant (α=0.05)",
                ],
            )

    def test_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "src/statistical_analysis.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
