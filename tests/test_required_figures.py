"""CPU unit and smoke tests for Section 16 required figure generation scripts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np
import torch
import yaml

from src import generate_qualitative_results as qr
from src import visualize_required_figures as vrf
from src.dataset import ACDCSegDataset, get_val_transforms
from src.motion import MotionEstimator
from src.segmentation_model import SegmentationUNet


def _make_anatomy_mask(height: int = 32, width: int = 32, radius_scale: float = 1.0) -> np.ndarray:
    """Create a synthetic cardiac mask with classes 0=BG, 1=LV, 2=MYO, 3=RV."""
    yy, xx = np.ogrid[:height, :width]
    r_lv2 = (4.5 * radius_scale) ** 2
    r_myo2 = (7.2 * radius_scale) ** 2
    r_rv2 = (3.5 * radius_scale) ** 2

    lv = ((yy - 16) ** 2 + (xx - 14) ** 2) <= r_lv2
    myo = (((yy - 16) ** 2 + (xx - 14) ** 2) <= r_myo2) & (~lv)
    rv = ((yy - 16) ** 2 + (xx - 22) ** 2) <= r_rv2

    mask = np.zeros((height, width), dtype=np.int64)
    mask[rv] = 3
    mask[myo] = 2
    mask[lv] = 1
    return mask


def _build_synthetic_workspace(root: Path) -> dict:
    """Populate a self-contained workspace for testing qualitative and required figure scripts."""
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = root / "checkpoints" / "experiments"
    motion_ckpt_dir = root / "checkpoints" / "motion"
    motion_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ssl_dir = root / "results" / "ssl"
    ssl_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = root / "results" / "pseudo_labels" / "100pct"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    train_pids = ["patient001", "patient002"]
    test_pids = ["patient003", "patient004"]
    (splits_dir / "train_patients.txt").write_text("\n".join(train_pids) + "\n", encoding="utf-8")
    (splits_dir / "test_patients.txt").write_text("\n".join(test_pids) + "\n", encoding="utf-8")

    rng = np.random.default_rng(42)
    index_entries = []
    pseudo_records = []

    ed_mask = _make_anatomy_mask(32, 32, radius_scale=1.0)
    es_mask = _make_anatomy_mask(32, 32, radius_scale=0.75)

    for pid in train_pids + test_pids:
        is_train = pid in train_pids
        for s_idx in range(3):
            scale_factors = [1.0, 0.93, 0.86, 0.80, 0.75] if is_train else [1.0, 0.75]
            frame_ids = [0, 1, 2, 3, 4] if is_train else [0, 4]
            phases = ["ED", "cine", "cine", "cine", "ES"] if is_train else ["ED", "ES"]

            for f_idx, phase, r_scale in zip(frame_ids, phases, scale_factors):
                anat = _make_anatomy_mask(32, 32, radius_scale=r_scale)
                base_img = (anat.astype(np.float32) * 0.45 + 0.1 * (s_idx + 1))
                noise = rng.normal(0.0, 0.05, size=(32, 32)).astype(np.float32)
                img = (base_img + noise).astype(np.float32)

                if phase == "ED":
                    mask = ed_mask.copy()
                    has_label = True
                elif phase == "ES":
                    mask = es_mask.copy()
                    has_label = True
                else:
                    mask = np.full((32, 32), -1, dtype=np.int64)
                    has_label = False

                fname = f"{pid}_frame{f_idx:02d}_slice{s_idx:02d}.npz"
                np.savez(
                    processed_dir / fname,
                    image=img,
                    mask=mask,
                    patient_id=pid,
                    frame_idx=f_idx,
                    slice_idx=s_idx,
                    has_label=has_label,
                    phase=phase,
                )
                index_entries.append(
                    {
                        "patient_id": pid,
                        "frame_idx": f_idx,
                        "slice_idx": s_idx,
                        "phase": phase,
                        "has_gt": has_label,
                        "file": fname,
                    }
                )

                if is_train and phase == "cine" and f_idx == 1 and s_idx == 1:
                    raw_pl = anat.copy()
                    conf = np.where(anat == 2, 0.72, 0.97).astype(np.float32)
                    accept = conf >= 0.90
                    filt_pl = np.where(accept, raw_pl, -1).astype(np.int64)
                    pl_fname = f"{pid}_frame{f_idx:02d}_slice{s_idx:02d}_pseudo.npz"
                    np.savez(
                        pseudo_dir / pl_fname,
                        raw_pseudo_label=raw_pl,
                        filtered_pseudo_label=filt_pl,
                        confidence=conf,
                        accept_mask=accept,
                        patient_id=pid,
                        slice_idx=s_idx,
                        frame_idx=f_idx,
                    )
                    pseudo_records.append(
                        {
                            "patient_id": pid,
                            "frame_idx": f_idx,
                            "slice_idx": s_idx,
                            "file": pl_fname,
                        }
                    )

    (processed_dir / "dataset_index.json").write_text(json.dumps(index_entries, indent=2), encoding="utf-8")
    (pseudo_dir / "pseudo_label_index.json").write_text(json.dumps(pseudo_records, indent=2), encoding="utf-8")

    # Save checkpoints: legacy supervised (no experiment_name) + tagged checkpoints + non-allowlisted bad run.
    runs = [
        ("supervised_10pct_seed42", False, True),
        ("supervised_100pct_seed42", True, True),
        ("full_pipeline_10pct_seed42", True, True),
        ("full_pipeline_100pct_seed42", True, True),
        ("patient999_untrusted", False, False),
    ]
    torch.manual_seed(7)
    for idx, (run_id, include_provenance, write_hist) in enumerate(runs):
        model = SegmentationUNet(in_channels=1, num_classes=4, encoder_channels=[8, 16, 32, 64])
        with torch.no_grad():
            for param in model.parameters():
                param.add_(idx * 0.004)
        run_dir = ckpt_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {"model_state_dict": model.state_dict()}
        if include_provenance:
            payload["experiment_name"] = run_id
        torch.save(payload, run_dir / f"{run_id}_best.pth")

        if write_hist:
            hist = {
                "train_loss": [0.95 - 0.1 * e for e in range(4)],
                "val_loss": [0.90 - 0.08 * e for e in range(4)],
                "val_dice": [0.40 + 0.12 * e for e in range(4)],
            }
            (run_dir / f"{run_id}_history.json").write_text(json.dumps(hist), encoding="utf-8")

    # Save trained synthetic MotionEstimator checkpoint with non-zero flow output.
    motion_model = MotionEstimator(channels=[16, 32, 64, 32], align_corners=True, padding_mode="border")
    with torch.no_grad():
        for p in motion_model.parameters():
            p.fill_(0.05)
    motion_ckpt = motion_ckpt_dir / "motion_model_best.pth"
    torch.save({"model_state_dict": motion_model.state_dict()}, motion_ckpt)

    ssl_hist = {
        "epoch": [1, 2, 3, 4],
        "total_loss": [0.45, 0.35, 0.28, 0.22],
        "val_total_loss": [0.48, 0.38, 0.31, 0.25],
        "recon_loss": [0.40, 0.30, 0.24, 0.19],
    }
    ssl_history_path = ssl_dir / "ssl_history.json"
    ssl_history_path.write_text(json.dumps(ssl_hist), encoding="utf-8")

    exp_cfg = {
        "data": {
            "processed_dir": str(processed_dir),
            "train_split": str(splits_dir / "train_patients.txt"),
            "test_split": str(splits_dir / "test_patients.txt"),
            "num_classes": 4,
        },
        "model": {
            "in_channels": 1,
            "num_classes": 4,
            "encoder_channels": [8, 16, 32, 64],
            "use_ssl_pretrained": False,
        },
        "logging": {
            "checkpoint_dir": str(ckpt_dir),
        },
    }
    motion_cfg = {
        "data": {
            "processed_dir": str(processed_dir),
            "train_split": str(splits_dir / "train_patients.txt"),
        },
        "motion_model": {"channels": [16, 32, 64, 32]},
        "warping": {"align_corners": True, "padding_mode": "border"},
        "loss": {"photometric_weight": 1.0, "smoothness_weight": 0.1},
    }
    pseudo_cfg = {
        "data": {
            "processed_dir": str(processed_dir),
            "train_split": str(splits_dir / "train_patients.txt"),
        },
        "filtering": {
            "confidence_metric": "max_probability",
            "confidence_threshold": 0.90,
            "entropy_threshold": 0.25,
            "use_temporal_consistency": True,
            "temporal_consistency_threshold": 0.80,
            "min_foreground_pixels": 5,
            "ignore_index": -1,
        },
        "model": {
            "in_channels": 1,
            "num_classes": 4,
            "encoder_channels": [8, 16, 32, 64],
            "checkpoint": str(ckpt_dir / "full_pipeline_100pct_seed42" / "full_pipeline_100pct_seed42_best.pth"),
            "motion_checkpoint": str(motion_ckpt),
        },
        "motion": {
            "channels": [16, 32, 64, 32],
            "align_corners": True,
            "padding_mode": "border",
        },
    }

    exp_cfg_path = root / "experiments.yaml"
    motion_cfg_path = root / "motion.yaml"
    pseudo_cfg_path = root / "pseudo_labels.yaml"
    exp_cfg_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")
    motion_cfg_path.write_text(yaml.safe_dump(motion_cfg), encoding="utf-8")
    pseudo_cfg_path.write_text(yaml.safe_dump(pseudo_cfg), encoding="utf-8")

    return {
        "processed_dir": processed_dir,
        "train_split": splits_dir / "train_patients.txt",
        "test_split": splits_dir / "test_patients.txt",
        "ckpt_dir": ckpt_dir,
        "motion_ckpt": motion_ckpt,
        "pseudo_dir": pseudo_dir,
        "ssl_history": ssl_history_path,
        "exp_cfg_path": exp_cfg_path,
        "motion_cfg_path": motion_cfg_path,
        "pseudo_cfg_path": pseudo_cfg_path,
        "exp_cfg": exp_cfg,
        "pseudo_cfg": pseudo_cfg,
    }


class RequiredFiguresTests(unittest.TestCase):
    def test_qualitative_compare_supervised_failure_cases_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = _build_synthetic_workspace(root)
            out_dir = root / "out_qual"

            # Canonical test split enforcement rejects synthetic 2-patient split when unpatched.
            dataset = ACDCSegDataset(
                str(env["processed_dir"]), str(env["test_split"]), transform=get_val_transforms()
            )
            self.assertEqual(len(dataset), 12)
            with self.assertRaises(ValueError):
                qr._verify_canonical_cohort(dataset)

            # Untrusted legacy checkpoint without experiment_name must still fail closed.
            with self.assertRaises(ValueError):
                qr.load_production_model(
                    "patient999_untrusted",
                    env["exp_cfg"],
                    torch.device("cpu"),
                    checkpoint_dir=str(env["ckpt_dir"]),
                )

            argv = [
                "generate_qualitative_results.py",
                "--config",
                str(env["exp_cfg_path"]),
                "--processed-dir",
                str(env["processed_dir"]),
                "--checkpoint-dir",
                str(env["ckpt_dir"]),
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
                "--compare-supervised",
                "--compare-fractions",
                "10",
                "100",
                "--failure-cases",
                "--max-failure-cases",
                "2",
            ]
            with patch.object(qr, "CANONICAL_TEST_SPLIT", env["test_split"]), patch.object(sys, "argv", argv):
                qr.main()

            comp_png = out_dir / "fig_baseline_vs_ssl.png"
            fail_png = out_dir / "fig_failure_cases.png"
            self.assertTrue(comp_png.exists() and comp_png.stat().st_size > 1024)
            self.assertTrue(fail_png.exists() and fail_png.stat().st_size > 1024)
            self.assertEqual(mpimg.imread(comp_png).ndim, 3)
            self.assertEqual(mpimg.imread(fail_png).ndim, 3)

    def test_visualize_required_figures_all_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = _build_synthetic_workspace(root)
            out_dir = root / "out_req"

            rc = vrf.main(
                [
                    "--all",
                    "--config",
                    str(env["exp_cfg_path"]),
                    "--motion-config",
                    str(env["motion_cfg_path"]),
                    "--pseudo-config",
                    str(env["pseudo_cfg_path"]),
                    "--processed-dir",
                    str(env["processed_dir"]),
                    "--checkpoint-dir",
                    str(env["ckpt_dir"]),
                    "--motion-checkpoint",
                    str(env["motion_ckpt"]),
                    "--pseudo-dir",
                    str(env["pseudo_dir"]),
                    "--ssl-history",
                    str(env["ssl_history"]),
                    "--output-dir",
                    str(out_dir),
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(rc, 0)

            for fname in [
                "fig_cine_motion.png",
                "fig_motion_flow.png",
                "fig_pseudo_filtering.png",
                "fig_training_curves.png",
            ]:
                path = out_dir / fname
                self.assertTrue(path.exists(), f"Expected output missing: {path}")
                self.assertGreater(path.stat().st_size, 1024)
                self.assertEqual(mpimg.imread(path).ndim, 3)

    def test_pseudo_filtering_live_teacher_path_on_unlabeled_cine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = _build_synthetic_workspace(root)
            out_dir = root / "out_live_pl"
            teacher_ckpt = env["ckpt_dir"] / "full_pipeline_100pct_seed42" / "full_pipeline_100pct_seed42_best.pth"

            out_path = vrf.generate_pseudo_filtering_figure(
                processed_dir=env["processed_dir"],
                pseudo_config=env["pseudo_cfg"],
                pseudo_dir=root / "nonexistent_pseudo_dir",
                teacher_checkpoint=str(teacher_ckpt),
                motion_checkpoint=str(env["motion_ckpt"]),
                device=torch.device("cpu"),
                output_dir=out_dir,
            )
            self.assertTrue(out_path.exists() and out_path.stat().st_size > 1024)

    def test_motion_flow_refuses_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = _build_synthetic_workspace(root)
            with self.assertRaises(FileNotFoundError):
                vrf.generate_motion_flow_figure(
                    processed_dir=env["processed_dir"],
                    split_file=str(env["train_split"]),
                    motion_checkpoint=root / "missing_motion.pth",
                    motion_config=yaml.safe_load(env["motion_cfg_path"].read_text(encoding="utf-8")),
                    device=torch.device("cpu"),
                    output_dir=root / "out_err",
                )

    def test_training_curves_skips_gracefully_when_histories_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out_curves"
            result = vrf.plot_training_curves(
                output_dir=out_dir,
                ssl_history_path=root / "missing_ssl_history.json",
                finetune_history_paths=[("100%", root / "missing_ft_history.json")],
            )
            self.assertIsNone(result)
            self.assertFalse((out_dir / "fig_training_curves.png").exists())

    def test_cli_help_flags(self):
        for script in ["src/generate_qualitative_results.py", "src/visualize_required_figures.py"]:
            proc = subprocess.run(
                [sys.executable, script, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{script} --help failed:\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
