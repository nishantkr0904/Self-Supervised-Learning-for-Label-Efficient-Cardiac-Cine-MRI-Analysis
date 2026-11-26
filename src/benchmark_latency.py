"""Benchmark model complexity and single-slice inference latency (Table 3).

Measures parameter counts, checkpoint footprints (on-disk ``.pth`` when found
and analytical FP32 weight size), and synchronized forward-pass latency
(``mean +/- std`` ms/slice) for all six Table 3 architectures using the
project's YAML configurations and canonical builders:

  1. CineMo-Net (SegmentationUNet from configs/experiments.yaml; excludes SimpleFlowNet)
  2. Supervised U-Net (SegmentationUNet from configs/baseline_config.yaml)
  3. SegFormer (MiT-B0 from configs/baseline_segformer.yaml)
  4. Swin-Unet (Swin-Tiny from configs/baseline_swin_unet.yaml)
  5. TransUNet (R50-ViT-B/16 from configs/baseline_transunet.yaml)
  6. MedSAM (ViT-B via src/medsam_baseline.py when segment_anything & weights exist)

Usage:
    python -m src.benchmark_latency --device auto --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medsam_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_ENV,
    build_medsam,
    default_checkpoint_path,
)
from src.segmentation_model import build_segmentation_model  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one Table 3 benchmark entry."""

    key: str
    method: str
    config_path: Optional[str]
    input_resolution: str
    checkpoint_candidates: tuple[str, ...]
    notes: str = ""


DEFAULT_OUTPUT_JSON = "results/tables/complexity_benchmark.json"

MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="cinemo_net",
        method="CineMo-Net",
        config_path="configs/experiments.yaml",
        input_resolution="256x256 (1ch)",
        checkpoint_candidates=(
            "checkpoints/experiments/full_pipeline_100pct_seed42/full_pipeline_100pct_seed42_best.pth",
            "/kaggle/working/checkpoints/experiments/full_pipeline_100pct_seed42/full_pipeline_100pct_seed42_best.pth",
        ),
        notes="Inference-only SegmentationUNet (excludes offline/training SimpleFlowNet).",
    ),
    ModelSpec(
        key="supervised_unet",
        method="Supervised U-Net",
        config_path="configs/baseline_config.yaml",
        input_resolution="256x256 (1ch)",
        checkpoint_candidates=(
            "checkpoints/experiments/supervised_100pct_seed42/supervised_100pct_seed42_best.pth",
            "checkpoints/baseline_unet_best.pth",
            "/kaggle/working/checkpoints/experiments/supervised_100pct_seed42/supervised_100pct_seed42_best.pth",
            "/kaggle/working/checkpoints/baseline_unet_best.pth",
        ),
        notes="Canonical 2D ResUNet trained from random initialization.",
    ),
    ModelSpec(
        key="segformer_mit_b0",
        method="SegFormer (MiT-B0)",
        config_path="configs/baseline_segformer.yaml",
        input_resolution="256x256 (1ch)",
        checkpoint_candidates=(
            "checkpoints/baseline_segformer_best.pth",
            "checkpoints/experiments/baseline_segformer_full/baseline_segformer_full_best.pth",
            "/kaggle/working/checkpoints/baseline_segformer_best.pth",
        ),
        notes="HuggingFace MiT-B0 encoder with 1ch stem adaptation and 4-class head.",
    ),
    ModelSpec(
        key="swin_unet_tiny",
        method="Swin-Unet (Swin-Tiny)",
        config_path="configs/baseline_swin_unet.yaml",
        input_resolution="256x256 (1ch)",
        checkpoint_candidates=(
            "checkpoints/baseline_swin_unet/baseline_swin_unet_best.pth",
            "checkpoints/baseline_swin_unet_best.pth",
            "checkpoints/experiments/baseline_swin_unet/baseline_swin_unet_best.pth",
            "/kaggle/working/checkpoints/baseline_swin_unet/baseline_swin_unet_best.pth",
        ),
        notes="Swin-Tiny hierarchical encoder with patch-expanding decoder.",
    ),
    ModelSpec(
        key="transunet_r50_vit_b16",
        method="TransUNet (R50-ViT-B/16)",
        config_path="configs/baseline_transunet.yaml",
        input_resolution="256x256 (1ch)",
        checkpoint_candidates=(
            "checkpoints/baseline_transunet_best.pth",
            "checkpoints/experiments/baseline_transunet_full/baseline_transunet_full_best.pth",
            "/kaggle/working/checkpoints/baseline_transunet_best.pth",
        ),
        notes="Hybrid ResNet-50 stem + 12-layer ViT-B/16 + CNN decoder.",
    ),
    ModelSpec(
        key="medsam_vit_b",
        method="MedSAM (ViT-B)",
        config_path=None,
        input_resolution="1024x1024 (3ch, 3 boxes)",
        checkpoint_candidates=(
            "checkpoints/medsam/medsam_vit_b.pth",
            "/kaggle/working/checkpoints/medsam/medsam_vit_b.pth",
        ),
        notes="Official ViT-B + prompt encoder + mask decoder via SamPredictor (3 class boxes).",
    ),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for complexity and latency benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark Table 3 complexity metrics (Params, Checkpoint Size, Latency)."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device for latency benchmarking (default: auto).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of untimed warmup forward passes per model (default: 20).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed forward passes per model (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size; reported latency is divided by batch size (default: 1).",
    )
    parser.add_argument(
        "--medsam-checkpoint",
        default=None,
        help="Optional explicit path to medsam_vit_b.pth.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_JSON}).",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve CLI --device flag to a concrete torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested via --device cuda, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def get_hardware_name(device: torch.device) -> str:
    """Return exact GPU name on CUDA or CPU host description."""
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device.index or 0))
    proc = platform.processor() or platform.machine() or "CPU"
    return f"CPU ({proc}, {os.cpu_count() or 1} cores)"


def format_size(num_bytes: int) -> str:
    """Format byte count using decimal SI units matching Table 3 (e.g. 24.3 MB, 1.18 GB)."""
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1e9:.2f} GB"
    return f"{num_bytes / 1e6:.1f} MB" if num_bytes >= 10_000_000 else f"{num_bytes / 1e6:.2f} MB"


def find_existing_checkpoint(
    candidates: Sequence[str],
    explicit_path: Optional[str] = None,
    include_medsam_defaults: bool = False,
) -> Optional[Path]:
    """Return an explicit override if valid, or the first existing candidate checkpoint."""
    if explicit_path:
        p = Path(explicit_path)
        return p if (p.exists() and p.is_file()) else None

    search_paths: List[Path] = []
    env_path = os.environ.get(DEFAULT_CHECKPOINT_ENV)
    if include_medsam_defaults and env_path:
        search_paths.append(Path(env_path))
    for item in candidates:
        p = Path(item)
        search_paths.append(p if p.is_absolute() else PROJECT_ROOT / p)
    if include_medsam_defaults:
        search_paths.append(default_checkpoint_path())
    for p in search_paths:
        if p.exists() and p.is_file():
            return p
    return None


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration relative to PROJECT_ROOT and validate it is a dict."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Missing config file: {full_path}")
    with full_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file {full_path}")
    return data


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters())


def time_forward_passes(
    forward_fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int,
) -> Dict[str, float]:
    """Measure synchronized per-slice inference latency (mean +/- std ms/slice)."""
    with torch.no_grad():
        for i in range(warmup):
            out = forward_fn()
            if i == 0 and tuple(out.shape) != (batch_size, 4, 256, 256):
                raise RuntimeError(
                    f"Unexpected output shape {tuple(out.shape)}; expected {(batch_size, 4, 256, 256)}."
                )
        if warmup == 0:
            out = forward_fn()
            if tuple(out.shape) != (batch_size, 4, 256, 256):
                raise RuntimeError(
                    f"Unexpected output shape {tuple(out.shape)}; expected {(batch_size, 4, 256, 256)}."
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        per_slice_ms: List[float] = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = forward_fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_slice_ms.append(elapsed_ms / float(batch_size))

    arr = np.asarray(per_slice_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1 if len(arr) > 1 else 0)),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _try_medsam_param_count() -> Optional[int]:
    """Compute exact MedSAM ViT-B parameter count when segment_anything is installed."""
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        return None
    sam = sam_model_registry["vit_b"](checkpoint=None)
    return count_parameters(sam)


def build_model_from_spec(
    spec: ModelSpec,
    device: torch.device,
    medsam_checkpoint: Optional[str] = None,
) -> tuple[Optional[nn.Module], Optional[Path], Optional[str]]:
    """Construct benchmark model from YAML/builder without mutating repository configs.

    For SegFormer, Swin-Unet, and TransUNet, ``pretrained`` is overridden to
    ``False`` for architecture instantiation so offline benchmarking measures the
    identical parameter count and forward latency without external downloads.
    """
    if spec.key == "medsam_vit_b":
        try:
            import segment_anything  # noqa: F401
        except ImportError:
            return None, None, "segment_anything is not installed"

        ckpt_path = find_existing_checkpoint(
            spec.checkpoint_candidates,
            explicit_path=medsam_checkpoint,
            include_medsam_defaults=True,
        )
        if ckpt_path is None:
            return (
                None,
                None,
                "MedSAM checkpoint (medsam_vit_b.pth) not found locally; pass --medsam-checkpoint",
            )
        model = build_medsam(checkpoint_path=str(ckpt_path), device=device)
        return model, ckpt_path, None

    cfg = copy.deepcopy(load_yaml_config(spec.config_path))
    if spec.key == "cinemo_net" and "full_pipeline" not in cfg.get("variants", {}):
        raise KeyError("Expected 'full_pipeline' variant in configs/experiments.yaml")

    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("architecture", "2d_unet") != "2d_unet":
        model_cfg["pretrained"] = False

    model = build_segmentation_model(cfg)
    ckpt_path = find_existing_checkpoint(spec.checkpoint_candidates)
    return model, ckpt_path, None


def benchmark_single_model(
    spec: ModelSpec,
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int,
    medsam_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark complexity and latency for a single model specification."""
    model, ckpt_path, skip_reason = build_model_from_spec(
        spec, device=device, medsam_checkpoint=medsam_checkpoint
    )

    if model is None:
        medsam_params = _try_medsam_param_count() if spec.key == "medsam_vit_b" else None
        fp32_bytes = medsam_params * 4 if medsam_params is not None else None
        return {
            "key": spec.key,
            "method": spec.method,
            "config_path": spec.config_path,
            "input_resolution": spec.input_resolution,
            "status": "skipped",
            "skip_reason": skip_reason,
            "params": medsam_params,
            "params_millions": round(medsam_params / 1e6, 2) if medsam_params is not None else None,
            "fp32_weights_bytes": fp32_bytes,
            "fp32_weights_display": format_size(fp32_bytes) if fp32_bytes is not None else None,
            "checkpoint_path": None,
            "checkpoint_disk_bytes": None,
            "checkpoint_display": "---",
            "latency_ms": None,
            "notes": spec.notes,
        }

    model = model.to(device)
    model.eval()

    total_params = count_parameters(model)
    fp32_bytes = total_params * 4
    disk_bytes = ckpt_path.stat().st_size if ckpt_path is not None else None
    ckpt_display = (
        format_size(disk_bytes)
        if disk_bytes is not None
        else f"{format_size(fp32_bytes)} (FP32)"
    )

    x = torch.randn(batch_size, 1, 256, 256, device=device, dtype=torch.float32)
    latency_stats = time_forward_passes(
        forward_fn=lambda: model(x),
        device=device,
        warmup=warmup,
        iters=iters,
        batch_size=batch_size,
    )

    if device.type == "cuda":
        del model, x
        torch.cuda.empty_cache()

    return {
        "key": spec.key,
        "method": spec.method,
        "config_path": spec.config_path,
        "input_resolution": spec.input_resolution,
        "status": "ok",
        "skip_reason": None,
        "params": total_params,
        "params_millions": round(total_params / 1e6, 2),
        "fp32_weights_bytes": fp32_bytes,
        "fp32_weights_display": format_size(fp32_bytes),
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
        "checkpoint_disk_bytes": disk_bytes,
        "checkpoint_display": ckpt_display,
        "latency_ms": latency_stats,
        "notes": spec.notes,
    }


def format_markdown_table(records: Sequence[Dict[str, Any]]) -> str:
    """Format benchmark results as a Markdown table matching Table 3."""
    lines = [
        "| Method | Params (M) | Checkpoint Size | Latency (ms/slice) |",
        "|---|---:|---:|---:|",
    ]
    for rec in records:
        params_str = f"{rec['params_millions']:.2f}" if rec["params_millions"] is not None else "---"
        ckpt_str = rec["checkpoint_display"] or "---"
        if rec["status"] == "ok" and rec["latency_ms"] is not None:
            lat = rec["latency_ms"]
            lat_str = f"{lat['mean_ms']:.2f} +/- {lat['std_ms']:.2f}"
        else:
            lat_str = "Skipped"
        lines.append(f"| {rec['method']} | {params_str} | {ckpt_str} | {lat_str} |")
    return "\n".join(lines)


def run_benchmark(
    device_arg: str = "auto",
    warmup: int = 20,
    iters: int = 100,
    batch_size: int = 1,
    medsam_checkpoint: Optional[str] = None,
    output_json: Optional[str] = DEFAULT_OUTPUT_JSON,
    model_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the Table 3 complexity & latency benchmark and optionally save JSON output."""
    if warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"--iters must be > 0, got {iters}")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {batch_size}")

    device = resolve_device(device_arg)
    hardware_name = get_hardware_name(device)
    selected = (
        [s for s in MODEL_SPECS if s.key in set(model_keys)]
        if model_keys is not None
        else list(MODEL_SPECS)
    )

    records: List[Dict[str, Any]] = []
    for spec in selected:
        rec = benchmark_single_model(
            spec,
            device=device,
            warmup=warmup,
            iters=iters,
            batch_size=batch_size,
            medsam_checkpoint=medsam_checkpoint,
        )
        if rec["status"] == "skipped":
            print(f"[SKIP] {rec['method']}: {rec['skip_reason']}")
        else:
            lat = rec["latency_ms"]
            print(
                f"[OK] {rec['method']}: {rec['params_millions']:.2f}M params | "
                f"{rec['checkpoint_display']} | "
                f"{lat['mean_ms']:.2f} +/- {lat['std_ms']:.2f} ms/slice on {hardware_name}"
            )
        records.append(rec)

    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_type": device.type,
        "hardware_name": hardware_name,
        "torch_version": torch.__version__,
        "warmup": warmup,
        "iters": iters,
        "batch_size": batch_size,
        "input_shape": [batch_size, 1, 256, 256],
        "models": records,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark JSON to {out_path}")

    print("\n" + format_markdown_table(records))
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_benchmark(
        device_arg=args.device,
        warmup=args.warmup,
        iters=args.iters,
        batch_size=args.batch_size,
        medsam_checkpoint=args.medsam_checkpoint,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
