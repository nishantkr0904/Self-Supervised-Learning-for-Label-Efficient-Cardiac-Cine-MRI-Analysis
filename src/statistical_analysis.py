"""Patient-level paired statistical analysis with Holm-Bonferroni correction.

Reads canonical n=20 per-patient test and perturbation robustness metrics from
``results/experiments/`` and computes two-sided paired Wilcoxon signed-rank tests
across all required comparison families without altering any model checkpoints or
experiment JSONs.

Outputs:
- ``results/tables/statistical_tests.json``
- ``results/tables/statistical_tests.csv``
- ``results/tables/statistical_tests.md``
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from scipy.stats import wilcoxon

CANONICAL_TEST_SPLIT = Path("data/splits/test_patients.txt")
METRICS = ("Mean_Dice", "Mean_HD95")


def _read_cohort_ids(split_file: Path) -> set:
    return {
        line.strip()
        for line in Path(split_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        corrected = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, corrected)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted


def load_test_patient_metrics(json_path: Path) -> Dict[str, Dict[str, float]]:
    """Load per-patient metric dict from a test_metrics.json file."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "per_patient" not in payload or not payload["per_patient"]:
        raise ValueError(f"Missing 'per_patient' metrics in {json_path}")
    return payload["per_patient"]


def load_robustness_patient_metrics(
    json_path: Path, condition_name: str
) -> Dict[str, Dict[str, float]]:
    """Load per-patient metric dict for a condition from robustness_metrics.json."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    conditions = payload.get("conditions", {})
    if condition_name not in conditions:
        raise KeyError(f"Condition '{condition_name}' not found in {json_path}")
    per_patient = conditions[condition_name].get("per_patient")
    if not per_patient:
        raise ValueError(f"Missing 'per_patient' under condition '{condition_name}' in {json_path}")
    return per_patient


def extract_matched_vectors(
    per_patient_a: Dict[str, Dict[str, float]],
    per_patient_b: Dict[str, Dict[str, float]],
    metric: str,
    expected_patients: Optional[set] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Align paired per-patient vectors and verify cohort completeness and finiteness."""
    keys_a = set(per_patient_a.keys())
    keys_b = set(per_patient_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"Patient cohort mismatch between compared runs: "
            f"only_a={sorted(keys_a - keys_b)}, only_b={sorted(keys_b - keys_a)}"
        )
    if expected_patients is not None and keys_a != expected_patients:
        raise ValueError(
            f"Evaluated patients ({len(keys_a)}) do not match canonical split "
            f"({len(expected_patients)}): missing={sorted(expected_patients - keys_a)}, "
            f"unexpected={sorted(keys_a - expected_patients)}"
        )

    patient_ids = sorted(keys_a)
    arr_a = np.array([float(per_patient_a[pid][metric]) for pid in patient_ids], dtype=np.float64)
    arr_b = np.array([float(per_patient_b[pid][metric]) for pid in patient_ids], dtype=np.float64)
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        raise ValueError(f"Non-finite values encountered for metric '{metric}'")
    return patient_ids, arr_a, arr_b


def compute_paired_wilcoxon(
    values_a: Sequence[float],
    values_b: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Run a two-sided paired Wilcoxon signed-rank test on aligned patient arrays.

    Uses ``ddof=0`` standard deviation to stay identical to ``src/metrics.py``.
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or arr_a.size == 0:
        raise ValueError("Aligned 1D non-empty vectors of equal length are required.")

    diffs = arr_a - arr_b
    if np.all(diffs == 0.0):
        w_stat = 0.0
        raw_p = 1.0
    else:
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        w_stat = float(res.statistic)
        raw_p = float(res.pvalue)

    return {
        "n_patients": int(arr_a.size),
        "mean_a": float(np.mean(arr_a)),
        "std_a": float(np.std(arr_a, ddof=0)),
        "mean_b": float(np.mean(arr_b)),
        "std_b": float(np.std(arr_b, ddof=0)),
        "paired_delta": float(np.mean(diffs)),
        "paired_delta_std": float(np.std(diffs, ddof=0)),
        "median_delta": float(np.median(diffs)),
        "w_stat": w_stat,
        "raw_p": raw_p,
        "alpha": float(alpha),
    }


def _format_mean_std(mean: float, std: float, metric: str) -> str:
    if metric == "Mean_HD95":
        return f"{mean:.2f} +/- {std:.2f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _format_delta(delta: float, metric: str) -> str:
    if metric == "Mean_HD95":
        return f"{delta:+.2f}"
    return f"{delta:+.4f}"


def _format_p(p_val: float) -> str:
    if p_val < 1e-4:
        return f"{p_val:.2e}"
    return f"{p_val:.4f}"


def build_comparison_specs(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Define all required paired comparison families (m=3, m=4, m=4, m=5)."""
    exp_dir = Path(experiments_dir)

    test_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    rob_cache: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}

    def get_test(run_id: str) -> Dict[str, Dict[str, float]]:
        if run_id not in test_cache:
            test_cache[run_id] = load_test_patient_metrics(exp_dir / run_id / "test_metrics.json")
        return test_cache[run_id]

    def get_rob(run_id: str, cond: str) -> Dict[str, Dict[str, float]]:
        key = (run_id, cond)
        if key not in rob_cache:
            rob_cache[key] = load_robustness_patient_metrics(
                exp_dir / run_id / "robustness_metrics.json", cond
            )
        return rob_cache[key]

    specs = [
        # 1. Label-Efficiency Family (m = 3)
        {
            "family": "Label-Efficiency (m=3)",
            "comparison": "Full Pipeline (10%) vs. Supervised (10%)",
            "label_a": "Full Pipeline (10%)",
            "label_b": "Supervised (10%)",
            "per_patient_a": get_test("full_pipeline_10pct_seed42"),
            "per_patient_b": get_test("supervised_10pct_seed42"),
        },
        {
            "family": "Label-Efficiency (m=3)",
            "comparison": "Full Pipeline (25%) vs. Supervised (25%)",
            "label_a": "Full Pipeline (25%)",
            "label_b": "Supervised (25%)",
            "per_patient_a": get_test("full_pipeline_25pct_seed42"),
            "per_patient_b": get_test("supervised_25pct_seed42"),
        },
        {
            "family": "Label-Efficiency (m=3)",
            "comparison": "Full Pipeline (50%) vs. Supervised (50%)",
            "label_a": "Full Pipeline (50%)",
            "label_b": "Supervised (50%)",
            "per_patient_a": get_test("full_pipeline_50pct_seed42"),
            "per_patient_b": get_test("supervised_50pct_seed42"),
        },
        # 2. Ablation Family (m = 4)
        {
            "family": "Ablation (m=4)",
            "comparison": "+Motion vs. +SSL (100%)",
            "label_a": "SSL + Motion (100%)",
            "label_b": "SSL Finetune (100%)",
            "per_patient_a": get_test("ssl_motion_100pct_seed42"),
            "per_patient_b": get_test("ssl_finetune_100pct_seed42"),
        },
        {
            "family": "Ablation (m=4)",
            "comparison": "+Pseudo (Raw) vs. +Motion (100%)",
            "label_a": "Full Pipeline Raw (100%)",
            "label_b": "SSL + Motion (100%)",
            "per_patient_a": get_test("full_pipeline_raw_100pct_seed42"),
            "per_patient_b": get_test("ssl_motion_100pct_seed42"),
        },
        {
            "family": "Ablation (m=4)",
            "comparison": "+Conf Filter (Full) vs. +Pseudo Raw (100%)",
            "label_a": "Full Pipeline Filtered (100%)",
            "label_b": "Full Pipeline Raw (100%)",
            "per_patient_a": get_test("full_pipeline_100pct_seed42"),
            "per_patient_b": get_test("full_pipeline_raw_100pct_seed42"),
        },
        {
            "family": "Ablation (m=4)",
            "comparison": "Alt. (SSL+Filtered) vs. Full Pipeline (100%)",
            "label_a": "SSL + Filtered Pseudo (100%)",
            "label_b": "Full Pipeline Filtered (100%)",
            "per_patient_a": get_test("ssl_pseudo_100pct_seed42"),
            "per_patient_b": get_test("full_pipeline_100pct_seed42"),
        },
        # 3. Robustness vs. Clean Family (m = 4)
        {
            "family": "Robustness vs. Clean (m=4)",
            "comparison": "Full Noise (std=0.05) vs. Clean",
            "label_a": "Full Noise (std=0.05)",
            "label_b": "Full Clean",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "gaussian_noise_0.05"),
            "per_patient_b": get_rob("full_pipeline_100pct_seed42", "clean"),
        },
        {
            "family": "Robustness vs. Clean (m=4)",
            "comparison": "Full Noise (std=0.10) vs. Clean",
            "label_a": "Full Noise (std=0.10)",
            "label_b": "Full Clean",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "gaussian_noise_0.10"),
            "per_patient_b": get_rob("full_pipeline_100pct_seed42", "clean"),
        },
        {
            "family": "Robustness vs. Clean (m=4)",
            "comparison": "Full Scale (0.9x) vs. Clean",
            "label_a": "Full Scale (0.9x)",
            "label_b": "Full Clean",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "intensity_scale_0.90"),
            "per_patient_b": get_rob("full_pipeline_100pct_seed42", "clean"),
        },
        {
            "family": "Robustness vs. Clean (m=4)",
            "comparison": "Full Scale (1.1x) vs. Clean",
            "label_a": "Full Scale (1.1x)",
            "label_b": "Full Clean",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "intensity_scale_1.10"),
            "per_patient_b": get_rob("full_pipeline_100pct_seed42", "clean"),
        },
        # 4. Robustness Full vs. Supervised (handoff) Family (m = 5)
        {
            "family": "Robustness Full vs. Sup. (m=5)",
            "comparison": "Clean: Full vs. Sup. (handoff)",
            "label_a": "Full Clean",
            "label_b": "Sup. Clean (handoff)",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "clean"),
            "per_patient_b": get_rob("supervised_100pct_seed42", "clean"),
        },
        {
            "family": "Robustness Full vs. Sup. (m=5)",
            "comparison": "Noise (0.05): Full vs. Sup. (handoff)",
            "label_a": "Full Noise (0.05)",
            "label_b": "Sup. Noise (0.05)",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "gaussian_noise_0.05"),
            "per_patient_b": get_rob("supervised_100pct_seed42", "gaussian_noise_0.05"),
        },
        {
            "family": "Robustness Full vs. Sup. (m=5)",
            "comparison": "Noise (0.10): Full vs. Sup. (handoff)",
            "label_a": "Full Noise (0.10)",
            "label_b": "Sup. Noise (0.10)",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "gaussian_noise_0.10"),
            "per_patient_b": get_rob("supervised_100pct_seed42", "gaussian_noise_0.10"),
        },
        {
            "family": "Robustness Full vs. Sup. (m=5)",
            "comparison": "Scale (0.9x): Full vs. Sup. (handoff)",
            "label_a": "Full Scale (0.9x)",
            "label_b": "Sup. Scale (0.9x)",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "intensity_scale_0.90"),
            "per_patient_b": get_rob("supervised_100pct_seed42", "intensity_scale_0.90"),
        },
        {
            "family": "Robustness Full vs. Sup. (m=5)",
            "comparison": "Scale (1.1x): Full vs. Sup. (handoff)",
            "label_a": "Full Scale (1.1x)",
            "label_b": "Sup. Scale (1.1x)",
            "per_patient_a": get_rob("full_pipeline_100pct_seed42", "intensity_scale_1.10"),
            "per_patient_b": get_rob("supervised_100pct_seed42", "intensity_scale_1.10"),
        },
    ]
    return specs


def run_statistical_analysis(
    experiments_dir: Path = Path("results/experiments"),
    output_dir: Path = Path("results/tables"),
    canonical_split: Optional[Path] = CANONICAL_TEST_SPLIT,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute paired Wilcoxon + Holm-Bonferroni tables and write JSON, CSV, and Markdown."""
    expected_patients = _read_cohort_ids(canonical_split) if canonical_split is not None else None
    specs = build_comparison_specs(experiments_dir)

    raw_rows: List[Dict[str, Any]] = []
    for metric in METRICS:
        for spec in specs:
            _, arr_a, arr_b = extract_matched_vectors(
                spec["per_patient_a"],
                spec["per_patient_b"],
                metric=metric,
                expected_patients=expected_patients,
            )
            stats = compute_paired_wilcoxon(arr_a, arr_b, alpha=alpha)
            raw_rows.append(
                {
                    "family": spec["family"],
                    "comparison": spec["comparison"],
                    "label_a": spec["label_a"],
                    "label_b": spec["label_b"],
                    "metric": metric,
                    **stats,
                }
            )

    # Apply Holm-Bonferroni step-down correction independently within each (family, metric) group.
    grouped_indices: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in enumerate(raw_rows):
        grouped_indices.setdefault((row["family"], row["metric"]), []).append(idx)

    for (_family, _metric), idxs in grouped_indices.items():
        adj = holm_bonferroni([raw_rows[i]["raw_p"] for i in idxs])
        for i, holm_p in zip(idxs, adj):
            raw_rows[i]["family_size_m"] = len(idxs)
            raw_rows[i]["holm_p"] = float(holm_p)
            raw_rows[i]["significant"] = bool(holm_p < alpha)

    # Order output chronologically by metric group (Mean_Dice first, then Mean_HD95) preserving family order.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "test": "two-sided paired Wilcoxon signed-rank test",
        "zero_method": "wilcox",
        "multiple_comparison_correction": "Holm-Bonferroni step-down within each comparison family per metric",
        "alpha": alpha,
        "n_patients": raw_rows[0]["n_patients"] if raw_rows else 0,
        "comparisons": raw_rows,
    }
    json_path = output_dir / "statistical_tests.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    csv_headers = [
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
    ]
    csv_path = output_dir / "statistical_tests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_headers)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(
                {
                    "Family": row["family"],
                    "Comparison": row["comparison"],
                    "Metric": row["metric"],
                    "Mean A ± Std A": _format_mean_std(row["mean_a"], row["std_a"], row["metric"]),
                    "Mean B ± Std B": _format_mean_std(row["mean_b"], row["std_b"], row["metric"]),
                    "Paired Δ": _format_delta(row["paired_delta"], row["metric"]),
                    "W-stat": f"{row['w_stat']:.1f}",
                    "Raw p": f"{row['raw_p']:.6g}",
                    "Holm p": f"{row['holm_p']:.6g}",
                    "Significant (α=0.05)": "Yes" if row["significant"] else "No",
                }
            )

    md_lines = [
        "# Section 17: Paired Statistical Analysis",
        "",
        (
            f"Two-sided paired Wilcoxon signed-rank tests (`zero_method=\"wilcox\"`, "
            f"`alpha={alpha:.2f}`) over `n={json_payload['n_patients']}` matched canonical test patients, "
            "with Holm-Bonferroni step-down correction within each comparison family per metric. "
            "`Paired Δ` is computed as `A - B` (`Mean_Dice` higher better; `Mean_HD95` negative means lower error)."
        ),
        "",
        "| " + " | ".join(csv_headers) + " |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in raw_rows:
        sig_str = "**Yes**" if row["significant"] else "No"
        md_lines.append(
            f"| {row['family']} | {row['comparison']} | `{row['metric']}` | "
            f"{_format_mean_std(row['mean_a'], row['std_a'], row['metric'])} | "
            f"{_format_mean_std(row['mean_b'], row['std_b'], row['metric'])} | "
            f"{_format_delta(row['paired_delta'], row['metric'])} | "
            f"{row['w_stat']:.1f} | {_format_p(row['raw_p'])} | {_format_p(row['holm_p'])} | {sig_str} |"
        )
    md_path = output_dir / "statistical_tests.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return json_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Section 17 paired Wilcoxon + Holm-Bonferroni statistical analysis."
    )
    parser.add_argument("--experiments-dir", type=str, default="results/experiments")
    parser.add_argument("--output-dir", type=str, default="results/tables")
    parser.add_argument("--canonical-split", type=str, default=str(CANONICAL_TEST_SPLIT))
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    canonical = Path(args.canonical_split) if args.canonical_split else None
    payload = run_statistical_analysis(
        experiments_dir=Path(args.experiments_dir),
        output_dir=Path(args.output_dir),
        canonical_split=canonical,
        alpha=args.alpha,
    )
    print(
        f"Wrote {Path(args.output_dir) / 'statistical_tests.{json,csv,md}'} "
        f"({len(payload['comparisons'])} tests across n={payload['n_patients']} patients)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
