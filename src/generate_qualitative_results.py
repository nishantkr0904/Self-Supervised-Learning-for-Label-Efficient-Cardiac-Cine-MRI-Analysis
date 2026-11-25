"""Qualitative-only test-set visualization for Full Pipeline runs.

Inference-only script: loads the existing production *_best.pth checkpoints,
runs a single forward pass on a small deterministic subset of the fixed
20-patient TEST split, and saves MRI / Ground Truth / Prediction / Overlay
figures. It performs no training, no metric computation, no checkpoint
selection, and never writes to results/experiments/, results/tables/,
checkpoints/, or data/splits/.

Usage:
    python -m src.generate_qualitative_results --all --device cuda
    python -m src.generate_qualitative_results --run-id full_pipeline_10pct_seed42 --device cuda
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import ACDCSegDataset, get_val_transforms
from src.evaluate_test import (
    CANONICAL_TEST_SPLIT as _EVAL_CANONICAL_TEST_SPLIT,
    LEGACY_ALLOWLIST_RUN_IDS,
    PRODUCTION_RUN_IDS,
    VOXEL_SPACING,
    _read_cohort_ids,
    build_model_from_config,
    parse_run_id,
    resolve_checkpoint as resolve_test_checkpoint,
    resolve_fraction_seed,
    validate_safe_run_id,
)
from src.metrics import compute_metrics_batch, compute_patient_level_metrics

CANONICAL_TEST_SPLIT = _EVAL_CANONICAL_TEST_SPLIT

LEGACY_QUALITATIVE_ALLOWLIST = (
    tuple(PRODUCTION_RUN_IDS)
    + tuple(LEGACY_ALLOWLIST_RUN_IDS)
    + ("supervised_100pct_seed42",)
)
# Keep backward-compatible alias for any external callers/tests.
LEGACY_SUPERVISED_QUALITATIVE_ALLOWLIST = LEGACY_QUALITATIVE_ALLOWLIST

# Discrete mask colors: 0 background (black), 1 LV red, 2 myocardium green, 3 RV blue.
MASK_CMAP = ListedColormap(["black", "#e41a1c", "#4daf4a", "#377eb8"])

# Deterministic example selection: sorted patient IDs at these positions,
# and the median foreground-bearing labeled slice per patient.
EXAMPLE_PATIENT_POSITIONS = (0, 10)


def resolve_checkpoint(
    run_id: str,
    checkpoint_dir: str = "checkpoints/experiments",
    checkpoint_path_override: Optional[str] = None,
) -> Path:
    """Return the checkpoint path for a run.

    Delegates to the shared evaluator resolver so both scripts accept the
    same layouts: the experiments ``<run_id>/<run_id>_best.pth`` layout by
    default, or an explicit ``*_best.pth`` file (e.g. train.py layout).
    """
    return resolve_test_checkpoint(run_id, checkpoint_dir, checkpoint_path_override)


def load_production_model(
    run_id: str, config: dict, device: torch.device,
    checkpoint_dir: str = "checkpoints/experiments",
    checkpoint_path_override: Optional[str] = None,
) -> torch.nn.Module:
    """Load a production or legacy-allowlisted checkpoint without modifying it."""
    checkpoint_path = resolve_checkpoint(run_id, checkpoint_dir, checkpoint_path_override)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    provenance = checkpoint.get("experiment_name")
    canonical_suffix = ("checkpoints", "experiments", run_id, f"{run_id}_best.pth")
    configured_canonical = (Path(checkpoint_dir) / run_id / f"{run_id}_best.pth").resolve()
    matches_canonical_layout = (
        checkpoint_path.resolve() == configured_canonical
        or checkpoint_path.resolve().parts[-4:] == canonical_suffix
    )
    if provenance is None:
        if run_id in LEGACY_QUALITATIVE_ALLOWLIST and matches_canonical_layout:
            print(
                f"Note: Allowing legacy checkpoint without 'experiment_name' "
                f"for allowlisted run '{run_id}': {checkpoint_path}"
            )
        else:
            raise ValueError(
                f"Refusing {checkpoint_path}: checkpoint provenance "
                f"'{provenance}' does not match requested run '{run_id}'. "
                "Only the run's own *_best.pth checkpoint may be visualized."
            )
    elif str(provenance) != run_id:
        raise ValueError(
            f"Refusing {checkpoint_path}: checkpoint provenance "
            f"'{provenance}' does not match requested run '{run_id}'. "
            "Only the run's own *_best.pth checkpoint may be visualized."
        )
    model = build_model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def _verify_canonical_cohort(dataset: ACDCSegDataset) -> None:
    """Ensure qualitative/failure figures operate only on the fixed 20-patient test split."""
    expected_cohort = _read_cohort_ids(CANONICAL_TEST_SPLIT)
    evaluated_cohort = set(dataset.get_patient_ids())
    if evaluated_cohort != expected_cohort:
        raise ValueError(
            f"Evaluated test cohort ({len(evaluated_cohort)} patients) does not match "
            f"canonical split {CANONICAL_TEST_SPLIT} ({len(expected_cohort)} patients)."
        )


def _compute_slice_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute unweighted foreground Mean Dice for a single slice."""
    res = compute_metrics_batch(
        np.expand_dims(pred, axis=0),
        np.expand_dims(gt, axis=0),
        patient_ids=["slice"],
        voxel_spacing=VOXEL_SPACING,
        compute_hd=False,
    )
    return float(res["per_sample"][0]["Mean_Dice"])


def _build_error_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return an RGB image highlighting TP (green), FP (red), and FN (cyan) foreground pixels."""
    gt_fg = gt > 0
    pred_fg = pred > 0
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    tp = gt_fg & pred_fg & (gt == pred)
    fp = pred_fg & (~gt_fg)
    fn = gt_fg & (~pred_fg)
    miscls = gt_fg & pred_fg & (gt != pred)
    rgb[tp] = [0.20, 0.78, 0.35]     # green: exact foreground match
    rgb[fp] = [0.92, 0.22, 0.22]     # red: false positive
    rgb[fn] = [0.15, 0.75, 0.95]     # cyan: missed ground-truth foreground
    rgb[miscls] = [0.95, 0.75, 0.10] # amber: class confusion within foreground
    return rgb


def select_examples(dataset: ACDCSegDataset) -> list:
    """Deterministically pick 2 patients x median foreground slice.

    Selection uses only patient/slice ordering and foreground presence —
    never test metrics or prediction quality.
    """
    patient_ids = sorted(dataset.get_patient_ids())
    if not patient_ids:
        return []
    if len(patient_ids) >= max(EXAMPLE_PATIENT_POSITIONS) + 1:
        chosen_pids = [patient_ids[i] for i in EXAMPLE_PATIENT_POSITIONS]
    else:
        chosen_pids = list(dict.fromkeys([patient_ids[0], patient_ids[-1]]))
    by_patient = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        by_patient.setdefault(str(sample["patient_id"]), []).append(idx)
    examples = []
    for pid in chosen_pids:
        indices = sorted(
            by_patient[pid], key=lambda i: int(dataset[i]["slice_idx"])
        )
        fg = [
            i for i in indices if bool((dataset[i]["mask"] > 0).any())
        ]
        pool = fg if fg else indices
        examples.append(pool[len(pool) // 2])
    return examples


@torch.no_grad()
def predict_indices(
    model: torch.nn.Module, dataset: ACDCSegDataset, indices: list, device: torch.device
) -> dict:
    """Run inference for the selected sample indices (no gradients)."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    wanted = set(indices)
    preds = {}
    for batch_idx, batch in enumerate(loader):
        base = batch_idx * loader.batch_size
        images = batch["image"].to(device)
        logits = model(images)
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        for j in range(batch_preds.shape[0]):
            if base + j in wanted:
                preds[base + j] = batch_preds[j]
    return preds


def render_panels(fig_axes, images, gts, preds, pids, title):
    """Fill a (n_examples x 4) axes grid: MRI | GT | Pred | Overlay."""
    for row, (img, gt, pred, pid) in enumerate(zip(images, gts, preds, pids)):
        fig_axes[row][0].imshow(img, cmap="gray")
        fig_axes[row][0].set_title("MRI" if row == 0 else "")
        fig_axes[row][0].set_ylabel(f"{pid}\nslice", rotation=0, labelpad=30,
                                    va="center", fontsize=9)
        fig_axes[row][1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=3)
        fig_axes[row][1].set_title("Ground Truth" if row == 0 else "")
        fig_axes[row][2].imshow(pred, cmap=MASK_CMAP, vmin=0, vmax=3)
        fig_axes[row][2].set_title("Prediction" if row == 0 else "")
        fig_axes[row][3].imshow(img, cmap="gray")
        fig_axes[row][3].imshow(
            np.ma.masked_where(pred == 0, pred),
            cmap=MASK_CMAP, vmin=0, vmax=3, alpha=0.55,
        )
        fig_axes[row][3].set_title("Overlay" if row == 0 else "")
        for ax in fig_axes[row]:
            ax.axis("off")
    fig_axes[0][0].figure.suptitle(title, fontsize=11)


def save_run_figure(run_id, label_fraction, images, gts, preds, pids, output_dir):
    fig, axes = plt.subplots(len(images), 4, figsize=(12, 4 * len(images)))
    render_panels(
        np.atleast_2d(axes), images, gts, preds, pids,
        f"{run_id} ({label_fraction}% labels) — TEST-set qualitative examples",
    )
    fig.tight_layout()
    path = Path(output_dir) / f"{run_id}_qualitative.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def save_comparison_figure(per_run_preds, images, gts, pids, output_dir):
    """Compact 2-row comparison across the four label fractions."""
    fractions = [10, 25, 50, 100]
    n_cols = 2 + len(fractions)
    fig, axes = plt.subplots(len(images), n_cols,
                             figsize=(3 * n_cols, 3.4 * len(images)))
    axes = np.atleast_2d(axes)
    headers = ["MRI", "Ground Truth"] + [f"Pred {f}%" for f in fractions]
    for row, (img, gt, pid) in enumerate(zip(images, gts, pids)):
        axes[row][0].imshow(img, cmap="gray")
        axes[row][1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=3)
        for col, frac in enumerate(fractions):
            run = f"full_pipeline_{frac}pct_seed42"
            axes[row][2 + col].imshow(per_run_preds[run][row],
                                      cmap=MASK_CMAP, vmin=0, vmax=3)
        axes[row][0].set_ylabel(f"{pid}\nslice", rotation=0, labelpad=30,
                                va="center", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=10)
    fig.suptitle("Full Pipeline TEST predictions across label fractions "
                 "(same deterministic examples)", fontsize=11)
    fig.tight_layout()
    path = Path(output_dir) / "final_qualitative_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def generate_baseline_vs_ssl_comparison(
    config: dict,
    dataset: ACDCSegDataset,
    device: torch.device,
    output_dir: Path,
    fractions=(10, 100),
    seed: int = 42,
    checkpoint_dir_override: Optional[str] = None,
) -> Path:
    """Compare Supervised Baseline vs Full Pipeline side-by-side on canonical test slices."""
    _verify_canonical_cohort(dataset)
    log_cfg = config.get("logging", {})
    checkpoint_dir = checkpoint_dir_override or log_cfg.get("checkpoint_dir", "checkpoints/experiments")
    example_indices = select_examples(dataset)

    images, gts, pids = [], [], []
    for idx in example_indices:
        sample = dataset[idx]
        images.append(sample["image"].squeeze(0).numpy())
        gts.append(sample["mask"].numpy())
        pids.append(str(sample["patient_id"]))

    n_rows = len(example_indices) * len(fractions)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    headers = [
        "MRI",
        "Ground Truth",
        "Supervised Baseline",
        "Full Pipeline (Ours)",
        "Full Pipeline Overlay",
    ]

    row_cursor = 0
    for frac in fractions:
        sup_id = f"supervised_{frac}pct_seed{seed}"
        full_id = f"full_pipeline_{frac}pct_seed{seed}"

        sup_model = load_production_model(sup_id, config, device, checkpoint_dir=checkpoint_dir)
        sup_preds = predict_indices(sup_model, dataset, example_indices, device)
        del sup_model

        full_model = load_production_model(full_id, config, device, checkpoint_dir=checkpoint_dir)
        full_preds = predict_indices(full_model, dataset, example_indices, device)
        del full_model

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for ex_idx, (img, gt, pid) in zip(example_indices, zip(images, gts, pids)):
            sup_pred = sup_preds[ex_idx]
            full_pred = full_preds[ex_idx]
            sup_dice = _compute_slice_dice(sup_pred, gt)
            full_dice = _compute_slice_dice(full_pred, gt)

            row_axes = axes[row_cursor]
            row_axes[0].imshow(img, cmap="gray")
            row_axes[0].set_ylabel(
                f"{pid}\n({frac}% labels)",
                rotation=0,
                labelpad=40,
                va="center",
                fontsize=9,
            )
            row_axes[1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=3)
            row_axes[2].imshow(sup_pred, cmap=MASK_CMAP, vmin=0, vmax=3)
            row_axes[2].set_xlabel(f"Dice: {sup_dice:.3f}", fontsize=8.5)
            row_axes[3].imshow(full_pred, cmap=MASK_CMAP, vmin=0, vmax=3)
            row_axes[3].set_xlabel(f"Dice: {full_dice:.3f}", fontsize=8.5)
            row_axes[4].imshow(img, cmap="gray")
            row_axes[4].imshow(
                np.ma.masked_where(full_pred == 0, full_pred),
                cmap=MASK_CMAP,
                vmin=0,
                vmax=3,
                alpha=0.55,
            )
            for ax in row_axes:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            row_cursor += 1

    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Qualitative Comparison: Supervised Baseline vs. Full Pipeline on Test Cohort",
        fontsize=11.5,
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "fig_baseline_vs_ssl.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


@torch.no_grad()
def generate_failure_cases_figure(
    config: dict,
    dataset: ACDCSegDataset,
    device: torch.device,
    output_dir: Path,
    run_id: str = "full_pipeline_100pct_seed42",
    max_cases: int = 3,
    batch_size: int = 8,
    checkpoint_dir_override: Optional[str] = None,
) -> Path:
    """Identify lowest-Dice / highest-HD95 test slices and render failure-analysis overlays."""
    _verify_canonical_cohort(dataset)
    log_cfg = config.get("logging", {})
    checkpoint_dir = checkpoint_dir_override or log_cfg.get("checkpoint_dir", "checkpoints/experiments")
    model = load_production_model(run_id, config, device, checkpoint_dir=checkpoint_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_images, all_targets, all_preds, all_pids, all_slices = [], [], [], [], []
    for batch in tqdm(loader, desc=f"Scanning failure cases ({run_id})", leave=False):
        imgs = batch["image"].to(device)
        preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
        all_images.append(batch["image"].squeeze(1).cpu().numpy())
        all_targets.append(batch["mask"].cpu().numpy())
        all_preds.append(preds)
        all_pids.extend([str(p) for p in batch["patient_id"]])
        all_slices.extend([int(s) for s in batch["slice_idx"]])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    images_np = np.concatenate(all_images, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    preds_np = np.concatenate(all_preds, axis=0)

    metrics = compute_metrics_batch(
        preds_np,
        targets_np,
        patient_ids=all_pids,
        voxel_spacing=VOXEL_SPACING,
        compute_hd=True,
    )
    per_sample = metrics["per_sample"]
    _ = compute_patient_level_metrics(per_sample)

    # Consider foreground-bearing ground-truth slices and select distinct-patient
    # outliers combining lowest Mean_Dice and highest Mean_HD95.
    fg_indices = [
        idx for idx in range(len(per_sample)) if bool((targets_np[idx] > 0).any())
    ]
    if not fg_indices:
        fg_indices = list(range(len(per_sample)))

    by_lowest_dice = sorted(fg_indices, key=lambda i: per_sample[i]["Mean_Dice"])
    by_highest_hd95 = sorted(fg_indices, key=lambda i: per_sample[i]["Mean_HD95"], reverse=True)

    selected = []
    seen_patients = set()
    for candidate_pool in (by_highest_hd95, by_lowest_dice):
        for idx in candidate_pool:
            pid = all_pids[idx]
            if idx not in selected and pid not in seen_patients:
                selected.append(idx)
                seen_patients.add(pid)
                break
    for idx in by_lowest_dice + by_highest_hd95:
        if len(selected) >= max_cases:
            break
        if idx not in selected:
            selected.append(idx)

    selected = selected[:max_cases]
    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3.4 * len(selected)))
    axes = np.atleast_2d(axes)
    headers = [
        "MRI",
        "Ground Truth",
        "Prediction",
        "Error Map (TP=G, FP=R, FN=C)",
    ]

    for row, idx in enumerate(selected):
        img = images_np[idx]
        gt = targets_np[idx]
        pred = preds_np[idx]
        m = per_sample[idx]
        err_rgb = _build_error_overlay(gt, pred)

        row_axes = axes[row]
        row_axes[0].imshow(img, cmap="gray")
        row_axes[0].set_ylabel(
            f"{all_pids[idx]}\nslice {all_slices[idx]:02d}",
            rotation=0,
            labelpad=38,
            va="center",
            fontsize=9,
        )
        row_axes[1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=3)
        row_axes[2].imshow(pred, cmap=MASK_CMAP, vmin=0, vmax=3)
        row_axes[2].set_xlabel(
            f"Dice: {m['Mean_Dice']:.3f} | HD95: {m['Mean_HD95']:.1f} mm",
            fontsize=8.5,
        )
        row_axes[3].imshow(img, cmap="gray")
        err_alpha = np.where((gt > 0) | (pred > 0), 0.75, 0.0)
        row_axes[3].imshow(err_rgb, alpha=err_alpha)
        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=9.8, fontweight="bold")

    fig.suptitle(
        f"Failure Cases & Boundary Outlier Analysis ({run_id})",
        fontsize=11.5,
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "fig_failure_cases.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative TEST visualizations for Full Pipeline runs"
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default="results/figures/qualitative")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Visualize a non-production run (e.g. a train.py baseline checkpoint). "
        "Must point at an existing '*_best.pth' file. Requires --label-fraction "
        "and --seed unless the run ID is a known production run.",
    )
    parser.add_argument(
        "--label-fraction",
        type=int,
        default=None,
        help="Label fraction recorded in figure titles (required with --checkpoint-path "
        "for non-production run IDs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed (required with --checkpoint-path for non-production run IDs).",
    )
    parser.add_argument(
        "--compare-supervised",
        action="store_true",
        help="Generate side-by-side Supervised Baseline vs Full Pipeline figure.",
    )
    parser.add_argument(
        "--compare-fractions",
        type=int,
        nargs="+",
        default=[10, 100],
        help="Label fractions to include in --compare-supervised (default: 10 100).",
    )
    parser.add_argument(
        "--compare-seed",
        type=int,
        default=42,
        help="Random seed suffix for --compare-supervised runs (default: 42).",
    )
    parser.add_argument(
        "--failure-cases",
        action="store_true",
        help="Generate lowest-Dice / highest-HD95 failure-case analysis figure.",
    )
    parser.add_argument(
        "--failure-run-id",
        type=str,
        default="full_pipeline_100pct_seed42",
        help="Run ID analyzed by --failure-cases (default: full_pipeline_100pct_seed42).",
    )
    parser.add_argument(
        "--max-failure-cases",
        type=int,
        default=3,
        help="Number of outlier slices plotted by --failure-cases (default: 3).",
    )
    args = parser.parse_args()

    if args.all:
        if args.checkpoint_path is not None:
            parser.error("--checkpoint-path cannot be combined with --all; visualize one --run-id at a time.")
        run_ids = list(PRODUCTION_RUN_IDS)
    elif args.run_id is not None:
        run_ids = [args.run_id]
    elif args.compare_supervised or args.failure_cases:
        run_ids = []
    else:
        parser.error("Provide --run-id <run_id>, --all, --compare-supervised, or --failure-cases.")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config.get("data", {})
    log_cfg = config.get("logging", {})
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    processed_dir = args.processed_dir or data_cfg.get("processed_dir", "data/processed")
    test_split = data_cfg.get("test_split", "data/splits/test_patients.txt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ACDCSegDataset(processed_dir, test_split, transform=get_val_transforms())
    if len(dataset) == 0:
        raise ValueError(f"Test dataset is empty: {test_split}")
    example_indices = select_examples(dataset)
    images, gts, pids = [], [], []
    for idx in example_indices:
        sample = dataset[idx]
        images.append(sample["image"].squeeze(0).numpy())
        gts.append(sample["mask"].numpy())
        pids.append(str(sample["patient_id"]))

    per_run_preds = {}
    for run_id in tqdm(run_ids, desc="Qualitative runs"):
        if args.checkpoint_path is None:
            label_fraction, _ = parse_run_id(run_id)
        else:
            validate_safe_run_id(run_id)
            label_fraction, _ = resolve_fraction_seed(
                run_id, args.label_fraction, args.seed
            )
        model = load_production_model(
            run_id, config, device,
            checkpoint_dir=args.checkpoint_dir or log_cfg.get("checkpoint_dir", "checkpoints/experiments"),
            checkpoint_path_override=args.checkpoint_path,
        )
        preds = predict_indices(model, dataset, example_indices, device)
        ordered = [preds[i] for i in example_indices]
        per_run_preds[run_id] = ordered
        save_run_figure(run_id, label_fraction, images, gts, ordered, pids,
                        output_dir)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.all and per_run_preds:
        save_comparison_figure(per_run_preds, images, gts, pids, output_dir)

    if args.compare_supervised:
        generate_baseline_vs_ssl_comparison(
            config=config,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            fractions=tuple(args.compare_fractions),
            seed=args.compare_seed,
            checkpoint_dir_override=args.checkpoint_dir,
        )

    if args.failure_cases:
        generate_failure_cases_figure(
            config=config,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            run_id=args.failure_run_id,
            max_cases=args.max_failure_cases,
            checkpoint_dir_override=args.checkpoint_dir,
        )


if __name__ == "__main__":
    main()
