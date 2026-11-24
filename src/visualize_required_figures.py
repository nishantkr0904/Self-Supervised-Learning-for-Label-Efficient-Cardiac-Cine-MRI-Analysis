"""Inference-only generator for Section 16 required figures on real data.

Produces publication-grade diagnostic visualizations using trained checkpoints
and preprocessed ACDC cine slices without fabricating synthetic values:
- fig_cine_motion.png: Multi-frame cardiac cine sequence across ED -> ES contraction
- fig_motion_flow.png: Trained MotionEstimator displacement field (magnitude + quiver),
  warped frame, and photometric residual reduction (fails if checkpoint is missing)
- fig_pseudo_filtering.png: Unlabeled intermediate cine slice before vs. after
  confidence + temporal-agreement filtering (ignore_index = -1 marked)
- fig_training_curves.png: SSL pretraining and/or fine-tuning loss/Dice curves from
  real saved history JSON files (gracefully skips with diagnostic note if absent)
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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
import yaml

from src.dataset import ACDCTemporalDataset
from src.experiment_runner import get_device, set_seed
from src.motion import MotionEstimator
from src.pseudo_labels import (
    PseudoLabelGenerator,
    UnlabeledTemporalPseudoDataset,
    load_motion_estimator,
    load_teacher_model,
)

# 0=BG (black), 1=LV (red), 2=MYO (green), 3=RV (blue)
MASK_CMAP = ListedColormap(["black", "#e41a1c", "#4daf4a", "#377eb8"])


def _index_patient_cine_series(
    processed_dir: Path,
    patient_id: Optional[str] = None,
    slice_idx: Optional[int] = None,
) -> Tuple[str, int, List[Path]]:
    """Find a real preprocessed cardiac cine sequence sorted along temporal frames."""
    dataset_index = processed_dir / "dataset_index.json"
    by_series: Dict[Tuple[str, int], List[Tuple[int, str, Path]]] = {}

    if dataset_index.exists():
        entries = json.loads(dataset_index.read_text(encoding="utf-8"))
        for entry in entries:
            pid = str(entry["patient_id"])
            if patient_id is not None and pid != patient_id:
                continue
            s_idx = int(entry["slice_idx"])
            if slice_idx is not None and s_idx != slice_idx:
                continue
            phase = str(entry.get("phase", "cine"))
            by_series.setdefault((pid, s_idx), []).append(
                (int(entry["frame_idx"]), phase, processed_dir / entry["file"])
            )
    else:
        for npz_path in sorted(processed_dir.glob("*.npz")):
            with np.load(npz_path, allow_pickle=True) as data:
                pid = str(data["patient_id"])
                if patient_id is not None and pid != patient_id:
                    continue
                s_idx = int(data["slice_idx"])
                if slice_idx is not None and s_idx != slice_idx:
                    continue
                phase = str(data["phase"]) if "phase" in data else "cine"
                f_idx = int(data["frame_idx"])
                by_series.setdefault((pid, s_idx), []).append((f_idx, phase, npz_path))

    if not by_series:
        raise FileNotFoundError(
            f"No preprocessed cine frames found in {processed_dir}"
            + (f" for patient={patient_id}" if patient_id else "")
        )

    # Prefer a mid-ventricular series that has multiple frames and ED/ES landmarks.
    def _series_score(key: Tuple[str, int]) -> Tuple[int, int, str, int]:
        frames = by_series[key]
        phases = {item[1] for item in frames}
        has_ed_es = int("ED" in phases and "ES" in phases)
        return (has_ed_es, len(frames), key[0], -key[1])

    best_key = max(sorted(by_series.keys()), key=_series_score)
    if patient_id is None and slice_idx is None:
        # Narrow to slices of the best patient and pick the middle slice.
        best_pid = best_key[0]
        pid_slices = sorted(
            {s for (p, s), frames in by_series.items() if p == best_pid and len(frames) >= 2}
        )
        if pid_slices:
            mid_slice = pid_slices[len(pid_slices) // 2]
            best_key = (best_pid, mid_slice)

    ordered = sorted(by_series[best_key], key=lambda item: item[0])
    return best_key[0], best_key[1], [item[2] for item in ordered]


def generate_cine_motion_figure(
    processed_dir: Path,
    output_dir: Path,
    patient_id: Optional[str] = None,
    slice_idx: Optional[int] = None,
    n_frames: int = 5,
) -> Path:
    """Plot consecutive/cycle ACDC cine frames showing cardiac contraction and change maps."""
    pid, s_idx, frame_paths = _index_patient_cine_series(
        processed_dir, patient_id=patient_id, slice_idx=slice_idx
    )
    if len(frame_paths) < 2:
        raise ValueError(f"Need >= 2 cine frames for {pid} slice {s_idx}; found {len(frame_paths)}.")

    # Select frames focusing on the contraction window [ED .. ES] if ED & ES are present.
    metas = []
    for path in frame_paths:
        with np.load(path, allow_pickle=True) as data:
            metas.append(
                {
                    "image": data["image"].astype(np.float32),
                    "frame_idx": int(data["frame_idx"]),
                    "phase": str(data["phase"]) if "phase" in data else "cine",
                }
            )

    ed_pos = next((i for i, m in enumerate(metas) if m["phase"] == "ED"), 0)
    es_pos = next((i for i, m in enumerate(metas) if m["phase"] == "ES"), len(metas) - 1)
    if es_pos > ed_pos and (es_pos - ed_pos + 1) >= min(n_frames, len(metas)):
        window = metas[ed_pos : es_pos + 1]
    else:
        window = metas

    count = min(n_frames, len(window))
    indices = np.linspace(0, len(window) - 1, count, dtype=int)
    chosen = [window[i] for i in indices]
    ref_img = chosen[0]["image"]

    vmax_diff = max(
        float(np.percentile(np.abs(m["image"] - ref_img), 99.5)) for m in chosen[1:]
    ) or 1e-3

    fig, axes = plt.subplots(2, count, figsize=(3.1 * count, 6.0))
    axes = np.atleast_2d(axes)

    for col, frame_data in enumerate(chosen):
        img = frame_data["image"]
        f_idx = frame_data["frame_idx"]
        phase = frame_data["phase"]
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"Frame {f_idx:02d} ({phase})", fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        diff = np.abs(img - ref_img)
        im_d = axes[1, col].imshow(diff, cmap="magma", vmin=0.0, vmax=vmax_diff)
        axes[1, col].set_title(
            "Reference" if col == 0 else f"|I_t - I_ref| (mean={diff.mean():.3f})",
            fontsize=8.8,
        )
        axes[1, col].axis("off")
        if col == count - 1:
            fig.colorbar(im_d, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"ACDC Cardiac Cine Motion Sequence ({pid}, Slice {s_idx:02d})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig_cine_motion.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def generate_motion_flow_figure(
    processed_dir: Path,
    split_file: str,
    motion_checkpoint: Path,
    motion_config: dict,
    device: torch.device,
    output_dir: Path,
) -> Path:
    """Visualize trained MotionEstimator flow vectors, warped frame, and residual reduction."""
    if not motion_checkpoint.exists():
        raise FileNotFoundError(
            f"Trained motion checkpoint is required for fig_motion_flow.png: {motion_checkpoint}"
        )

    dataset = ACDCTemporalDataset(str(processed_dir), split_file)
    if len(dataset) == 0:
        raise ValueError(f"No adjacent temporal pairs found in {processed_dir} for {split_file}")

    # Pick an active adjacent pair (maximum intensity motion near middle slices).
    best_idx = 0
    best_diff = -1.0
    scan_limit = min(len(dataset), 250)
    for idx in range(scan_limit):
        sample = dataset[idx]
        diff_score = float(torch.abs(sample["frame_t1"] - sample["frame_t"]).mean().item())
        if diff_score > best_diff:
            best_diff = diff_score
            best_idx = idx

    sample = dataset[best_idx]
    frame_t = sample["frame_t"].unsqueeze(0).to(device)
    frame_t1 = sample["frame_t1"].unsqueeze(0).to(device)

    model_cfg = motion_config.get("motion_model", {})
    warp_cfg = motion_config.get("warping", {})
    loss_cfg = motion_config.get("loss", {})
    motion_model = MotionEstimator(
        channels=model_cfg.get("channels", [16, 32, 64, 32]),
        align_corners=warp_cfg.get("align_corners", True),
        padding_mode=warp_cfg.get("padding_mode", "border"),
        photometric_weight=loss_cfg.get("photometric_weight", 1.0),
        smoothness_weight=loss_cfg.get("smoothness_weight", 0.1),
    ).to(device)
    ckpt = torch.load(motion_checkpoint, map_location=device, weights_only=False)
    motion_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    motion_model.eval()

    with torch.no_grad():
        out = motion_model(frame_t, frame_t1)

    img_t = frame_t.squeeze().cpu().numpy()
    img_t1 = frame_t1.squeeze().cpu().numpy()
    warped = out["warped_t"].squeeze().cpu().numpy()
    flow = out["flow"].squeeze().cpu().numpy()  # (2, H, W)
    dx, dy = flow[0], flow[1]
    flow_mag = np.sqrt(dx ** 2 + dy ** 2)

    raw_diff = np.abs(img_t1 - img_t)
    comp_diff = np.abs(img_t1 - warped)
    raw_mean = float(raw_diff.mean())
    comp_mean = float(comp_diff.mean())

    print(
        f"Motion visualization on {sample['patient_id']} slice {sample['slice_idx']} "
        f"(t={sample['frame_idx_t']}->{sample['frame_idx_t1']}): "
        f"max_flow={flow_mag.max():.3f} px, raw_L1={raw_mean:.4f}, warped_L1={comp_mean:.4f}"
    )

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.3))
    axes[0].imshow(img_t, cmap="gray")
    axes[0].set_title(f"Frame t={sample['frame_idx_t']:02d}", fontsize=10.5)
    axes[0].axis("off")

    axes[1].imshow(img_t1, cmap="gray")
    axes[1].set_title(f"Frame t+1={sample['frame_idx_t1']:02d}", fontsize=10.5)
    axes[1].axis("off")

    im_mag = axes[2].imshow(flow_mag, cmap="viridis")
    step = max(8, flow_mag.shape[0] // 20)
    yy, xx = np.mgrid[0 : flow_mag.shape[0] : step, 0 : flow_mag.shape[1] : step]
    axes[2].quiver(
        xx,
        yy,
        dx[yy, xx],
        dy[yy, xx],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.004,
        alpha=0.85,
    )
    axes[2].set_title(f"Displacement Field |phi|\n(Max: {flow_mag.max():.2f} px)", fontsize=10.5)
    axes[2].axis("off")
    fig.colorbar(im_mag, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(warped, cmap="gray")
    axes[3].set_title("Warped W(I_t, phi)", fontsize=10.5)
    axes[3].axis("off")

    err_vmax = max(float(np.percentile(raw_diff, 99.5)), float(np.percentile(comp_diff, 99.5)), 1e-4)
    im_err = axes[4].imshow(comp_diff, cmap="inferno", vmin=0.0, vmax=err_vmax)
    axes[4].set_title(
        f"Warped Residual |I_{{t+1}} - W(I_t)|\n(Mean {comp_mean:.4f} vs Raw {raw_mean:.4f})",
        fontsize=10.0,
    )
    axes[4].axis("off")
    fig.colorbar(im_err, ax=axes[4], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Trained Motion Estimator Visualization ({sample['patient_id']}, Slice {sample['slice_idx']:02d})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig_motion_flow.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def _render_pseudo_filtering_panel(
    image: np.ndarray,
    raw_label: np.ndarray,
    filtered_label: np.ndarray,
    confidence: np.ndarray,
    accept_mask: np.ndarray,
    title_suffix: str,
    threshold: float,
    output_dir: Path,
) -> Path:
    """Render MRI | Raw Teacher Argmax | Confidence Map | Confidence-Filtered Mask."""
    acc_pct = 100.0 * float(accept_mask.mean())
    rej_pct = 100.0 - acc_pct

    # Render filtered pseudo-labels with ignored (-1) pixels highlighted in light gray
    rgb_filtered = np.zeros((*filtered_label.shape, 3), dtype=np.float32)
    palette = {
        0: [0.05, 0.05, 0.05],  # accepted background
        1: [0.89, 0.10, 0.11],  # LV red
        2: [0.30, 0.69, 0.29],  # MYO green
        3: [0.22, 0.49, 0.72],  # RV blue
    }
    for cls_id, color in palette.items():
        rgb_filtered[filtered_label == cls_id] = color
    rejected = filtered_label == -1
    rgb_filtered[rejected] = [0.85, 0.85, 0.85]  # light gray for ignore_index=-1

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Unlabeled Cine MRI\n({title_suffix})", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(raw_label, cmap=MASK_CMAP, vmin=0, vmax=3)
    axes[1].set_title("Raw Pseudo-Label\n(100.0% pixels, unfiltered)", fontsize=10)
    axes[1].axis("off")

    im_conf = axes[2].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Confidence x Temp. Agree.\n(Threshold tau >= {threshold:.2f})", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im_conf, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(rgb_filtered)
    axes[3].set_title(
        f"Filtered Pseudo-Label\n({acc_pct:.2f}% accepted, {rej_pct:.2f}% rejected)",
        fontsize=10,
    )
    axes[3].axis("off")
    legend_handles = [
        Patch(facecolor=palette[1], label="LV"),
        Patch(facecolor=palette[2], label="MYO"),
        Patch(facecolor=palette[3], label="RV"),
        Patch(facecolor=[0.85, 0.85, 0.85], edgecolor="gray", label="Ignored (-1)"),
    ]
    axes[3].legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "Pseudo-Label Confidence & Motion-Consistency Filtering (Intermediate Unlabeled Frame)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig_pseudo_filtering.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def generate_pseudo_filtering_figure(
    processed_dir: Path,
    pseudo_config: dict,
    pseudo_dir: Optional[Path],
    teacher_checkpoint: Optional[str],
    motion_checkpoint: Optional[str],
    device: torch.device,
    output_dir: Path,
) -> Path:
    """Visualize raw vs confidence-filtered pseudo-labels on a real unlabeled cine slice."""
    filter_cfg = pseudo_config.get("filtering", {})
    threshold = float(filter_cfg.get("confidence_threshold", 0.90))

    # Path A: Use existing pre-generated offline pseudo-label artifacts if present.
    manifest_path = pseudo_dir / "pseudo_label_index.json" if pseudo_dir is not None else None
    if manifest_path is not None and manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        chosen_record = None
        best_score = -1.0
        for rec in records[: min(len(records), 400)]:
            npz_file = pseudo_dir / rec["file"]
            if not npz_file.exists():
                continue
            with np.load(npz_file, allow_pickle=True) as pl_data:
                raw_lbl = pl_data["raw_pseudo_label"]
                acc = pl_data["accept_mask"]
                fg_frac = float((raw_lbl > 0).mean())
                acc_rate = float(acc.mean())
                if fg_frac > 0.005:
                    # Prefer slices showing visible foreground and non-trivial rejection boundaries.
                    rej_bonus = min(1.0 - acc_rate, 0.25)
                    score = fg_frac + rej_bonus
                    if score > best_score:
                        best_score = score
                        chosen_record = rec
        if chosen_record is not None:
            pid = str(chosen_record["patient_id"])
            f_idx = int(chosen_record["frame_idx"])
            s_idx = int(chosen_record["slice_idx"])
            img_path = processed_dir / f"{pid}_frame{f_idx:02d}_slice{s_idx:02d}.npz"
            with np.load(img_path, allow_pickle=True) as img_data:
                if "mask" in img_data and not np.all(img_data["mask"] == -1):
                    raise ValueError(f"Expected unlabeled cine frame (mask == -1) at {img_path}")
                image = img_data["image"].astype(np.float32)
            with np.load(pseudo_dir / chosen_record["file"], allow_pickle=True) as pl_data:
                return _render_pseudo_filtering_panel(
                    image=image,
                    raw_label=pl_data["raw_pseudo_label"].astype(np.int64),
                    filtered_label=pl_data["filtered_pseudo_label"].astype(np.int64),
                    confidence=pl_data["confidence"].astype(np.float32),
                    accept_mask=pl_data["accept_mask"].astype(bool),
                    title_suffix=f"{pid}, f={f_idx:02d}, s={s_idx:02d}",
                    threshold=threshold,
                    output_dir=output_dir,
                )

    # Path B: Compute live on an unlabeled intermediate training slice using teacher + motion checkpoints.
    if not teacher_checkpoint:
        raise FileNotFoundError(
            f"No pre-generated pseudo-label manifest at {manifest_path} and no --teacher-checkpoint provided."
        )

    cfg_copy = json.loads(json.dumps(pseudo_config))
    cfg_copy["data"]["processed_dir"] = str(processed_dir)
    cfg_copy["model"]["checkpoint"] = str(teacher_checkpoint)
    if motion_checkpoint is not None:
        cfg_copy["model"]["motion_checkpoint"] = str(motion_checkpoint)

    dataset = UnlabeledTemporalPseudoDataset(
        str(processed_dir),
        cfg_copy["data"]["train_split"],
    )
    if len(dataset) == 0:
        raise ValueError(f"No unlabeled intermediate training slices found in {processed_dir}.")

    teacher = load_teacher_model(cfg_copy, device)
    motion_est = load_motion_estimator(cfg_copy, device)
    generator = PseudoLabelGenerator(
        model=teacher,
        device=device,
        confidence_metric=filter_cfg.get("confidence_metric", "max_probability"),
        confidence_threshold=threshold,
        entropy_threshold=float(filter_cfg.get("entropy_threshold", 0.25)),
        use_temporal_consistency=bool(filter_cfg.get("use_temporal_consistency", True)),
        temporal_consistency_threshold=float(filter_cfg.get("temporal_consistency_threshold", 0.80)),
        ignore_index=int(filter_cfg.get("ignore_index", -1)),
        min_foreground_pixels=int(filter_cfg.get("min_foreground_pixels", 30)),
        motion_estimator=motion_est,
    )

    best_payload = None
    best_score = -1.0
    scan_limit = min(len(dataset), 120)
    for idx in range(scan_limit):
        sample = dataset[idx]
        batch = {
            "frame_t": sample["frame_t"].unsqueeze(0),
            "frame_t1": sample["frame_t1"].unsqueeze(0),
        }
        out = generator.process_batch(batch)
        raw_np = out["raw_pseudo_labels"][0].cpu().numpy()
        acc_np = out["accept_mask"][0].cpu().numpy().astype(bool)
        fg_frac = float((raw_np > 0).mean())
        acc_rate = float(acc_np.mean())
        score = fg_frac + min(1.0 - acc_rate, 0.25)
        if score > best_score:
            best_score = score
            best_payload = (
                sample["frame_t"].squeeze(0).numpy(),
                raw_np,
                out["filtered_labels"][0].cpu().numpy(),
                out["confidence"][0].cpu().numpy(),
                acc_np,
                f"{sample['patient_id']}, f={sample['frame_idx_t']:02d}, s={sample['slice_idx']:02d}",
            )

    if best_payload is None:
        raise RuntimeError("Failed to generate pseudo-label visualization from dataset.")

    img_np, raw_np, filt_np, conf_np, acc_np, suffix = best_payload
    return _render_pseudo_filtering_panel(
        image=img_np,
        raw_label=raw_np,
        filtered_label=filt_np,
        confidence=conf_np,
        accept_mask=acc_np,
        title_suffix=suffix,
        threshold=threshold,
        output_dir=output_dir,
    )


def plot_training_curves(
    output_dir: Path,
    ssl_history_path: Optional[Path] = None,
    finetune_history_paths: Optional[Sequence[Tuple[str, Path]]] = None,
) -> Optional[Path]:
    """Plot real SSL pretraining and fine-tuning loss/Dice curves from history JSONs.

    Returns the output path when at least one valid history file exists; otherwise
    prints a clear message and returns None without fabricating synthetic curves.
    """
    ssl_data = None
    if ssl_history_path is not None and ssl_history_path.exists():
        ssl_data = json.loads(ssl_history_path.read_text(encoding="utf-8"))

    ft_histories: List[Tuple[str, Dict[str, Any]]] = []
    for label, path in (finetune_history_paths or []):
        if path.exists():
            ft_histories.append((label, json.loads(path.read_text(encoding="utf-8"))))

    if ssl_data is None and not ft_histories:
        print(
            "Note: Skipping fig_training_curves.png — no SSL or fine-tuning history JSON files found "
            f"(checked ssl={ssl_history_path}, finetune={[str(p) for _, p in (finetune_history_paths or [])]})."
        )
        return None

    panels = []
    if ssl_data is not None:
        panels.append("ssl")
    if ft_histories:
        panels.extend(["ft_loss", "ft_dice"])

    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.3))
    if len(panels) == 1:
        axes = [axes]

    ax_idx = 0
    if ssl_data is not None:
        ax = axes[ax_idx]
        epochs = ssl_data.get("epoch", list(range(1, len(ssl_data.get("total_loss", [])) + 1)))
        if "total_loss" in ssl_data:
            ax.plot(epochs, ssl_data["total_loss"], label="Train Total Loss", lw=2)
        if "val_total_loss" in ssl_data:
            ax.plot(epochs, ssl_data["val_total_loss"], "--", label="Val Total Loss", lw=1.8)
        if "recon_loss" in ssl_data:
            ax.plot(epochs, ssl_data["recon_loss"], ":", label="Train Recon Loss", lw=1.5)
        ax.set_title("SSL Pretraining Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8.5)
        ax_idx += 1

    if ft_histories:
        ax_loss = axes[ax_idx]
        ax_dice = axes[ax_idx + 1]
        for label, hist in ft_histories:
            train_loss = hist.get("train_loss", [])
            val_loss = hist.get("val_loss", [])
            val_dice = hist.get("val_dice", [])
            ep_loss = list(range(1, len(train_loss) + 1))
            ax_loss.plot(ep_loss, train_loss, label=f"{label} (Train)", lw=1.8)
            if val_loss:
                ax_loss.plot(list(range(1, len(val_loss) + 1)), val_loss, "--", label=f"{label} (Val)", lw=1.5)
            if val_dice:
                ep_dice = list(range(1, len(val_dice) + 1))
                ax_dice.plot(ep_dice, val_dice, label=label, lw=2.0)

        ax_loss.set_title("Fine-Tuning Loss vs. Epoch", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Dice + CE Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.legend(fontsize=8)

        ax_dice.set_title("Fine-Tuning Validation Mean Dice", fontsize=11, fontweight="bold")
        ax_dice.set_xlabel("Epoch")
        ax_dice.set_ylabel("Validation Mean Dice")
        ax_dice.grid(True, linestyle="--", alpha=0.4)
        ax_dice.legend(fontsize=8.5)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig_training_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Section 16 required figures from real data and checkpoints."
    )
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--motion-config", type=str, default="configs/motion.yaml")
    parser.add_argument("--pseudo-config", type=str, default="configs/pseudo_labels.yaml")
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/experiments")
    parser.add_argument("--motion-checkpoint", type=str, default="checkpoints/motion/motion_model_best.pth")
    parser.add_argument("--pseudo-dir", type=str, default="results/pseudo_labels/100pct")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--ssl-history", type=str, default="results/ssl/ssl_history.json")
    parser.add_argument(
        "--finetune-history",
        action="append",
        dest="finetune_histories",
        default=None,
        help="Optional history JSON path(s) for fine-tuning curves (repeatable).",
    )
    parser.add_argument("--output-dir", type=str, default="results/figures/required")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--slice-idx", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Generate all Section 16 visual figures.")
    parser.add_argument("--cine-motion", action="store_true", help="Generate fig_cine_motion.png.")
    parser.add_argument("--motion-flow", action="store_true", help="Generate fig_motion_flow.png.")
    parser.add_argument("--pseudo-filtering", action="store_true", help="Generate fig_pseudo_filtering.png.")
    parser.add_argument("--training-curves", action="store_true", help="Generate fig_training_curves.png if histories exist.")
    args = parser.parse_args(argv)

    if not any([args.all, args.cine_motion, args.motion_flow, args.pseudo_filtering, args.training_curves]):
        parser.error("Specify --all or at least one figure flag (--cine-motion, --motion-flow, --pseudo-filtering, --training-curves).")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    motion_cfg = yaml.safe_load(Path(args.motion_config).read_text(encoding="utf-8"))
    pseudo_cfg = yaml.safe_load(Path(args.pseudo_config).read_text(encoding="utf-8"))

    processed_dir = Path(args.processed_dir or exp_cfg["data"]["processed_dir"])
    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)

    if args.all or args.cine_motion:
        generate_cine_motion_figure(
            processed_dir=processed_dir,
            output_dir=output_dir,
            patient_id=args.patient_id,
            slice_idx=args.slice_idx,
        )

    if args.all or args.motion_flow:
        train_split = motion_cfg["data"].get("train_split", "data/splits/train_patients.txt")
        generate_motion_flow_figure(
            processed_dir=processed_dir,
            split_file=train_split,
            motion_checkpoint=Path(args.motion_checkpoint),
            motion_config=motion_cfg,
            device=device,
            output_dir=output_dir,
        )

    if args.all or args.pseudo_filtering:
        generate_pseudo_filtering_figure(
            processed_dir=processed_dir,
            pseudo_config=pseudo_cfg,
            pseudo_dir=Path(args.pseudo_dir) if args.pseudo_dir else None,
            teacher_checkpoint=args.teacher_checkpoint,
            motion_checkpoint=args.motion_checkpoint,
            device=device,
            output_dir=output_dir,
        )

    if args.all or args.training_curves:
        ft_pairs: List[Tuple[str, Path]] = []
        if args.finetune_histories:
            for path_str in args.finetune_histories:
                p = Path(path_str)
                ft_pairs.append((p.stem.replace("_history", ""), p))
        else:
            for run_id, label in [
                ("full_pipeline_100pct_seed42", "Full Pipeline (100%)"),
                ("full_pipeline_10pct_seed42", "Full Pipeline (10%)"),
                ("supervised_100pct_seed42", "Supervised Baseline (100%)"),
            ]:
                ft_pairs.append((label, ckpt_dir / run_id / f"{run_id}_history.json"))

        plot_training_curves(
            output_dir=output_dir,
            ssl_history_path=Path(args.ssl_history) if args.ssl_history else None,
            finetune_history_paths=ft_pairs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
