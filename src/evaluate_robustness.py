"""Inference-only test-time perturbation robustness evaluator (Step 12).

Evaluates trained 2D cardiac segmentation checkpoints on the canonical test
split under deterministic test-time image perturbations defined in
``configs/experiments.yaml``:
- Clean baseline (noise_std=0.0, intensity_scale=1.0)
- Additive zero-mean Gaussian noise (e.g. std in {0.05, 0.10})
- Multiplicative intensity scaling (e.g. scale in {0.9, 1.1})

Reuses the existing evaluation stack (`src/evaluate_test.py`, `src/dataset.py`,
`src/metrics.py`) without altering any training configs or existing test results.
"""

import os
import sys

# Ensure project root is on sys.path and prevent src/ from shadowing stdlib modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import ACDCSegDataset, get_val_transforms
from src.evaluate_test import (
    CANONICAL_TEST_SPLIT as _EVAL_CANONICAL_TEST_SPLIT,
    VOXEL_SPACING,
    _read_cohort_ids,
    build_model_from_config,
    resolve_checkpoint,
    validate_safe_run_id,
)
from src.experiment_runner import get_device, set_seed
from src.metrics import compute_metrics_batch, compute_patient_level_metrics

CANONICAL_TEST_SPLIT = _EVAL_CANONICAL_TEST_SPLIT

DEFAULT_RUN_IDS = [
    "supervised_100pct_seed42",
    "full_pipeline_100pct_seed42",
]

RUN_DISPLAY_NAMES = {
    "supervised_100pct_seed42": "Supervised Baseline (100%)",
    "unet_100pct_seed42": "Supervised Baseline (100%)",
    "full_pipeline_100pct_seed42": "Full Pipeline (Ours, 100%)",
    "full_pipeline_raw_100pct_seed42": "Full Pipeline Raw (100%)",
    "ssl_pseudo_100pct_seed42": "SSL + Filtered Pseudo (100%)",
    "ssl_motion_100pct_seed42": "SSL + Motion (100%)",
    "ssl_finetune_100pct_seed42": "SSL Only (100%)",
}

METRIC_KEYS = [
    "LV_Dice",
    "Myocardium_Dice",
    "RV_Dice",
    "Mean_Dice",
    "LV_HD95",
    "Myocardium_HD95",
    "RV_HD95",
    "Mean_HD95",
]


def build_perturbation_conditions(config: dict) -> List[Dict[str, Any]]:
    """Build ordered, deduplicated test-time perturbation conditions from config."""
    rob_cfg = config.get("robustness", {})
    noise_levels = [float(x) for x in rob_cfg.get("noise_levels", [0.0, 0.05, 0.10])]
    intensity_scalings = [float(x) for x in rob_cfg.get("intensity_scalings", [0.9, 1.0, 1.1])]

    conditions: List[Dict[str, Any]] = [
        {
            "name": "clean",
            "label": "Standard (Clean)",
            "noise_std": 0.0,
            "intensity_scale": 1.0,
        }
    ]
    seen = {(0.0, 1.0)}

    for sigma in noise_levels:
        if sigma < 0.0:
            raise ValueError(f"noise_levels must be non-negative; got {sigma}")
        key = (round(sigma, 8), 1.0)
        if key not in seen and sigma > 0.0:
            seen.add(key)
            conditions.append(
                {
                    "name": f"gaussian_noise_{sigma:.2f}",
                    "label": f"Gaussian Noise (std={sigma:.2f})",
                    "noise_std": sigma,
                    "intensity_scale": 1.0,
                }
            )

    for scale in intensity_scalings:
        if scale <= 0.0:
            raise ValueError(f"intensity_scalings must be positive; got {scale}")
        key = (0.0, round(scale, 8))
        if key not in seen:
            seen.add(key)
            conditions.append(
                {
                    "name": f"intensity_scale_{scale:.2f}",
                    "label": f"Intensity Scale ({scale:g}x)",
                    "noise_std": 0.0,
                    "intensity_scale": scale,
                }
            )

    return conditions


def apply_perturbation(
    images: torch.Tensor,
    noise_std: float = 0.0,
    intensity_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply multiplicative intensity scaling and additive zero-mean Gaussian noise.

    Images are kept unclamped because preprocessed ACDC slices are per-volume
    Z-score normalized (mean ~ 0, std ~ 1) with unbounded signed intensities.
    """
    perturbed = images * float(intensity_scale) if intensity_scale != 1.0 else images.clone()
    if noise_std > 0.0:
        gen_device = generator.device if generator is not None else torch.device("cpu")
        noise = torch.randn(
            images.shape,
            generator=generator,
            device=gen_device,
            dtype=torch.float32,
        )
        perturbed = perturbed + noise.to(device=images.device, dtype=images.dtype) * float(noise_std)
    return perturbed


def load_checkpoint_model(
    run_id: str,
    config: dict,
    device: torch.device,
    checkpoint_dir: str,
    checkpoint_path_override: Optional[str] = None,
) -> Tuple[torch.nn.Module, Path]:
    """Load a segmentation checkpoint with strict experiment_name provenance validation."""
    safe_run_id = validate_safe_run_id(run_id)
    ckpt_path = resolve_checkpoint(
        safe_run_id,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path_override=checkpoint_path_override,
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    provenance = checkpoint.get("experiment_name") if isinstance(checkpoint, dict) else None
    if provenance is None or str(provenance) != safe_run_id:
        raise ValueError(
            f"Checkpoint provenance mismatch for run_id '{safe_run_id}': "
            f"expected experiment_name={safe_run_id!r}, got {provenance!r} ({ckpt_path})"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {ckpt_path}")

    model = build_model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt_path


def collect_test_batches(
    processed_dir: str,
    test_split: str,
    batch_size: int = 16,
    canonical_test_split: Optional[Path] = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, List[str]]], int, int]:
    """Load and validate the canonical test split once into sequential CPU batches."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    test_dataset = ACDCSegDataset(
        processed_dir=processed_dir,
        split_file=test_split,
        transform=get_val_transforms(),
    )
    if not len(test_dataset):
        raise ValueError(f"Test dataset is empty for split: {test_split}")

    canonical_path = canonical_test_split or CANONICAL_TEST_SPLIT
    evaluated_cohort = set(test_dataset.get_patient_ids())
    canonical_cohort = _read_cohort_ids(canonical_path)
    if evaluated_cohort != canonical_cohort:
        raise ValueError(
            f"Evaluated test cohort ({len(evaluated_cohort)} patients) does not match "
            f"canonical split {canonical_path} ({len(canonical_cohort)} patients)."
        )

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batches: List[Tuple[torch.Tensor, torch.Tensor, List[str]]] = []
    total_slices = 0
    for batch in loader:
        images = batch["image"].cpu()
        masks = batch["mask"].cpu()
        patient_ids = [str(pid) for pid in batch["patient_id"]]
        total_slices += len(patient_ids)
        batches.append((images, masks, patient_ids))

    return batches, len(evaluated_cohort), total_slices


def evaluate_condition(
    model: torch.nn.Module,
    cached_batches: List[Tuple[torch.Tensor, torch.Tensor, List[str]]],
    condition: Dict[str, Any],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Run deterministic inference for a single perturbation condition.

    Resets a CPU torch.Generator to ``seed`` before iterating batches so that:
    1) Results are reproducible across CPU and CUDA executions.
    2) Varying Gaussian noise levels draw from identical paired standard normal noise.
    """
    model.eval()
    noise_std = float(condition["noise_std"])
    intensity_scale = float(condition["intensity_scale"])
    generator = torch.Generator(device="cpu").manual_seed(seed) if noise_std > 0.0 else None

    per_sample_metrics: List[Dict[str, Any]] = []
    with torch.no_grad():
        for images, masks, patient_ids in cached_batches:
            perturbed = apply_perturbation(
                images,
                noise_std=noise_std,
                intensity_scale=intensity_scale,
                generator=generator,
            ).to(device)
            logits = model(perturbed)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_results = compute_metrics_batch(
                preds,
                masks.numpy(),
                patient_ids=patient_ids,
                voxel_spacing=VOXEL_SPACING,
                compute_hd=True,
            )
            per_sample_metrics.extend(batch_results["per_sample"])

    aggregated = compute_patient_level_metrics(per_sample_metrics, compute_hd=True)
    return {
        "name": condition["name"],
        "label": condition["label"],
        "noise_std": noise_std,
        "intensity_scale": intensity_scale,
        "n_patients": int(aggregated["n_patients"]),
        "n_slices": len(per_sample_metrics),
        "mean": {k: float(v) for k, v in aggregated["mean"].items()},
        "std": {k: float(v) for k, v in aggregated["std"].items()},
        "per_patient": {
            pid: {mk: float(mv) for mk, mv in pmetrics.items()}
            for pid, pmetrics in sorted(aggregated["per_patient"].items())
        },
    }


def evaluate_run_robustness(
    run_id: str,
    config: dict,
    device: torch.device,
    seed: int = 42,
    batch_size: int = 16,
    processed_dir_override: Optional[str] = None,
    checkpoint_dir_override: Optional[str] = None,
    checkpoint_path_override: Optional[str] = None,
    results_dir_override: Optional[str] = None,
    cached_test_data: Optional[Tuple[List[Tuple[torch.Tensor, torch.Tensor, List[str]]], int, int]] = None,
    canonical_test_split: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a trained checkpoint across all robustness perturbations and save JSON."""
    data_cfg = config["data"]
    log_cfg = config.get("logging", {})
    processed_dir = processed_dir_override or data_cfg["processed_dir"]
    test_split = data_cfg.get("test_split", str(canonical_test_split or CANONICAL_TEST_SPLIT))
    checkpoint_dir = checkpoint_dir_override or log_cfg.get("checkpoint_dir", "checkpoints/experiments")
    results_root = Path(results_dir_override or log_cfg.get("results_dir", "results/experiments"))

    if cached_test_data is None:
        cached_batches, n_patients, n_slices = collect_test_batches(
            processed_dir=processed_dir,
            test_split=test_split,
            batch_size=batch_size,
            canonical_test_split=canonical_test_split,
        )
    else:
        cached_batches, n_patients, n_slices = cached_test_data

    model, ckpt_path = load_checkpoint_model(
        run_id=run_id,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path_override=checkpoint_path_override,
    )

    conditions = build_perturbation_conditions(config)
    condition_results: Dict[str, Dict[str, Any]] = {}
    clean_mean: Optional[Dict[str, float]] = None

    for cond in conditions:
        res = evaluate_condition(
            model=model,
            cached_batches=cached_batches,
            condition=cond,
            device=device,
            seed=seed,
        )
        if cond["name"] == "clean":
            clean_mean = res["mean"]
        res["delta_vs_clean"] = {
            key: float(res["mean"][key] - clean_mean[key]) if clean_mean is not None and key in clean_mean else 0.0
            for key in res["mean"]
        }
        condition_results[cond["name"]] = res

    report = {
        "run_id": run_id,
        "display_name": RUN_DISPLAY_NAMES.get(run_id, run_id),
        "checkpoint": str(ckpt_path),
        "test_split": str(test_split),
        "processed_dir": str(processed_dir),
        "n_patients": n_patients,
        "n_slices": n_slices,
        "voxel_spacing": list(VOXEL_SPACING),
        "seed": int(seed),
        "device": str(device),
        "conditions": condition_results,
    }

    run_results_dir = results_root / validate_safe_run_id(run_id)
    run_results_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_results_dir / "robustness_metrics.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _format_summary_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten condition metrics across evaluated runs into tabular rows."""
    rows: List[Dict[str, Any]] = []
    for report in reports:
        run_id = report["run_id"]
        variant = report.get("display_name", RUN_DISPLAY_NAMES.get(run_id, run_id))
        for cond in report["conditions"].values():
            mean = cond["mean"]
            std = cond["std"]
            delta = cond["delta_vs_clean"]
            rows.append(
                {
                    "Perturbation / Condition": cond["label"],
                    "Target Variant": variant,
                    "Run ID": run_id,
                    "Noise Std": cond["noise_std"],
                    "Intensity Scale": cond["intensity_scale"],
                    "Mean Dice": mean["Mean_Dice"],
                    "Mean Dice Std": std["Mean_Dice"],
                    "LV Dice": mean["LV_Dice"],
                    "MYO Dice": mean["Myocardium_Dice"],
                    "RV Dice": mean["RV_Dice"],
                    "HD95 (mm)": mean["Mean_HD95"],
                    "HD95 Std (mm)": std["Mean_HD95"],
                    "Delta Mean Dice": delta["Mean_Dice"],
                    "Status": "Evaluated",
                }
            )
    return rows


def write_robustness_tables(reports: List[Dict[str, Any]], tables_dir: Path) -> Tuple[Path, Path]:
    """Write aggregated robustness_table.csv and robustness_table.md."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = _format_summary_rows(reports)

    csv_path = tables_dir / "robustness_table.csv"
    csv_fields = [
        "Perturbation / Condition",
        "Target Variant",
        "Run ID",
        "Noise Std",
        "Intensity Scale",
        "Mean Dice",
        "Mean Dice Std",
        "LV Dice",
        "MYO Dice",
        "RV Dice",
        "HD95 (mm)",
        "HD95 Std (mm)",
        "Delta Mean Dice",
        "Status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "Noise Std": f"{row['Noise Std']:.2f}",
                    "Intensity Scale": f"{row['Intensity Scale']:g}",
                    "Mean Dice": f"{row['Mean Dice']:.4f}",
                    "Mean Dice Std": f"{row['Mean Dice Std']:.4f}",
                    "LV Dice": f"{row['LV Dice']:.4f}",
                    "MYO Dice": f"{row['MYO Dice']:.4f}",
                    "RV Dice": f"{row['RV Dice']:.4f}",
                    "HD95 (mm)": f"{row['HD95 (mm)']:.2f}",
                    "HD95 Std (mm)": f"{row['HD95 Std (mm)']:.2f}",
                    "Delta Mean Dice": f"{row['Delta Mean Dice']:+.4f}",
                }
            )

    md_path = tables_dir / "robustness_table.md"
    first = reports[0] if reports else {}
    lines = [
        "# Test-Time Perturbation Robustness Analysis (Step 12)",
        "",
        (
            f"Evaluated on canonical test cohort (`{first.get('n_patients', 0)}` patients, "
            f"`{first.get('n_slices', 0)}` slices, spacing `{tuple(first.get('voxel_spacing', VOXEL_SPACING))}`, "
            f"seed `{first.get('seed', 42)}`)."
        ),
        "",
        "| Perturbation / Condition | Target Variant | Mean Dice | LV Dice | MYO Dice | RV Dice | HD95 (mm) | Delta Mean Dice | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for row in rows:
        lines.append(
            "| {cond} | {variant} | {md:.4f} +/- {mds:.4f} | {lv:.4f} | {myo:.4f} | {rv:.4f} | {hd:.2f} +/- {hds:.2f} | {delta:+.4f} | {status} |".format(
                cond=row["Perturbation / Condition"],
                variant=row["Target Variant"],
                md=row["Mean Dice"],
                mds=row["Mean Dice Std"],
                lv=row["LV Dice"],
                myo=row["MYO Dice"],
                rv=row["RV Dice"],
                hd=row["HD95 (mm)"],
                hds=row["HD95 Std (mm)"],
                delta=row["Delta Mean Dice"],
                status=row["Status"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def print_robustness_summary(reports: List[Dict[str, Any]]) -> None:
    """Print a compact summary table of perturbation robustness results to stdout."""
    header = (
        f"{'Run ID':<30} | {'Condition':<26} | {'Mean Dice':>17} | "
        f"{'LV':>6} | {'MYO':>6} | {'RV':>6} | {'HD95(mm)':>13} | {'Delta':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for report in reports:
        run_id = report["run_id"]
        for cond in report["conditions"].values():
            mean = cond["mean"]
            std = cond["std"]
            delta = cond["delta_vs_clean"]["Mean_Dice"]
            dice_str = f"{mean['Mean_Dice']:.4f} +/- {std['Mean_Dice']:.4f}"
            hd_str = f"{mean['Mean_HD95']:.2f} +/- {std['Mean_HD95']:.2f}"
            print(
                f"{run_id:<30} | {cond['label']:<26} | {dice_str:>17} | "
                f"{mean['LV_Dice']:.4f} | {mean['Myocardium_Dice']:.4f} | {mean['RV_Dice']:.4f} | "
                f"{hd_str:>13} | {delta:>+7.4f}"
            )
    print("=" * len(header) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained segmentation checkpoints under deterministic test-time perturbations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments.yaml",
        help="Path to experiment YAML configuration (default: configs/experiments.yaml).",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        default=None,
        help="Run ID to evaluate (repeatable; e.g. --run-id supervised_100pct_seed42).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate default comparison runs: supervised_100pct_seed42 and full_pipeline_100pct_seed42.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device preference (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic test-time Gaussian noise (default: 42).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Override data.processed_dir (useful for Kaggle input datasets).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override logging.checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit checkpoint file path (only valid when evaluating a single --run-id).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override logging.results_dir for saving robustness_metrics.json.",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=None,
        help="Override logging.tables_dir when --write-table is enabled.",
    )
    parser.add_argument(
        "--write-table",
        action="store_true",
        help="Write/update results/tables/robustness_table.csv and robustness_table.md.",
    )
    args = parser.parse_args(argv)

    selected_runs: List[str] = []
    if args.all:
        selected_runs.extend(DEFAULT_RUN_IDS)
    if args.run_ids:
        selected_runs.extend(args.run_ids)
    args.resolved_run_ids = list(dict.fromkeys(selected_runs))

    if not args.resolved_run_ids:
        parser.error("Provide at least one --run-id or pass --all.")
    if args.checkpoint_path is not None and len(args.resolved_run_ids) > 1:
        parser.error("--checkpoint-path can only be used with a single --run-id.")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config["data"]
    log_cfg = config.get("logging", {})
    processed_dir = args.processed_dir or data_cfg["processed_dir"]
    test_split = data_cfg.get("test_split", str(CANONICAL_TEST_SPLIT))

    # Load and validate the canonical test cohort once across all requested runs.
    cached_test_data = collect_test_batches(
        processed_dir=processed_dir,
        test_split=test_split,
        batch_size=args.batch_size,
    )

    reports: List[Dict[str, Any]] = []
    for run_id in args.resolved_run_ids:
        report = evaluate_run_robustness(
            run_id=run_id,
            config=config,
            device=device,
            seed=args.seed,
            batch_size=args.batch_size,
            processed_dir_override=processed_dir,
            checkpoint_dir_override=args.checkpoint_dir,
            checkpoint_path_override=args.checkpoint_path,
            results_dir_override=args.results_dir,
            cached_test_data=cached_test_data,
        )
        reports.append(report)

    print_robustness_summary(reports)

    if args.write_table:
        tables_dir = Path(args.tables_dir or log_cfg.get("tables_dir", "results/tables"))
        csv_path, md_path = write_robustness_tables(reports, tables_dir)
        print(f"Wrote robustness tables: {csv_path} and {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
