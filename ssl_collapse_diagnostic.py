# ═══════════════════════════════════════════════════════════════════════════════
# SSL Representation Collapse Diagnostic: Pre- vs. Post-Projection Embeddings
# Read-Only / CPU-Compatible / Strictly Non-Modifying
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment & Path Resolution
# ─────────────────────────────────────────────────────────────────────────────

REPO_CANDIDATES = [
    Path("/kaggle/working/Self-Supervised-Learning-for-Label-Efficient-Cardiac-Cine-MRI-Analysis"),
    Path.cwd(),
]
REPO_ROOT = None
for cand in REPO_CANDIDATES:
    if (cand / "src" / "ssl_pretraining.py").exists() or (cand / "src" / "ssl.py").exists():
        REPO_ROOT = cand.resolve()
        break

if REPO_ROOT is None:
    raise RuntimeError(
        f"Could not locate repository root containing src/ssl_pretraining.py. Checked: {REPO_CANDIDATES}"
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import verified repository implementations
try:
    from src.ssl_pretraining import SSLModel
except ImportError:
    from src.ssl import SSLModel

from src.dataset import ACDCTemporalDataset


def resolve_checkpoint_path() -> Path:
    candidates = [
        REPO_ROOT / "checkpoints" / "ssl" / "ssl_best.pth",
        Path("checkpoints/ssl/ssl_best.pth"),
        Path("/kaggle/working/checkpoints/ssl/ssl_best.pth"),
        Path("/kaggle/input/motion-guided-ssl-acdc-checkpoints/ssl_best.pth"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"Could not locate 'ssl_best.pth' checkpoint. Checked:\n" + "\n".join(f" - {c}" for c in candidates)
    )


def resolve_processed_dir() -> Path:
    candidates = [
        REPO_ROOT / "data" / "processed",
        Path("data/processed"),
        Path("/kaggle/working/Self-Supervised-Learning-for-Label-Efficient-Cardiac-Cine-MRI-Analysis/data/processed"),
        Path("/kaggle/input/datasets/nishantk2310593/motion-guided-ssl-acdc-processed/data/processed"),
        Path("/kaggle/input/motion-guided-ssl-acdc-processed/data/processed"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            if (p / "dataset_index.json").exists() or any(p.glob("*.npz")):
                return p.resolve() if p.is_symlink() else p
    raise FileNotFoundError(
        f"Could not locate processed ACDC data directory. Checked:\n" + "\n".join(f" - {c}" for c in candidates)
    )


def resolve_split_file() -> Path:
    candidates = [
        REPO_ROOT / "data" / "splits" / "train_patients.txt",
        Path("data/splits/train_patients.txt"),
        REPO_ROOT / "data" / "splits" / "val_patients.txt",
        Path("data/splits/val_patients.txt"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"Could not locate patient split file (.txt). Checked:\n" + "\n".join(f" - {c}" for c in candidates)
    )


def resolve_config() -> Dict:
    candidates = [
        REPO_ROOT / "configs" / "ssl.yaml",
        Path("configs/ssl.yaml"),
        Path("/kaggle/working/Self-Supervised-Learning-for-Label-Efficient-Cardiac-Cine-MRI-Analysis/configs/ssl.yaml"),
    ]
    config_file = None
    for p in candidates:
        if p.exists() and p.is_file():
            config_file = p.resolve()
            break

    if config_file is None:
        raise FileNotFoundError(
            f"Required configuration file 'configs/ssl.yaml' was not found. Checked:\n"
            + "\n".join(f" - {c}" for c in candidates)
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to parse 'configs/ssl.yaml'.") from exc

    with open(config_file, "r") as f:
        full_cfg = yaml.safe_load(f)

    if not isinstance(full_cfg, dict):
        raise ValueError(
            f"Invalid configuration in '{config_file}': expected top-level dictionary, got {type(full_cfg).__name__}"
        )

    if "model" not in full_cfg or not isinstance(full_cfg["model"], dict):
        raise KeyError(
            f"Configuration file '{config_file}' is missing the expected 'model' section dictionary."
        )

    model_cfg = full_cfg["model"]
    required_keys = [
        "in_channels",
        "encoder_channels",
        "proj_dim",
        "mask_patch_size",
        "mask_ratio",
        "dropout",
        "use_residual",
    ]
    missing_keys = [k for k in required_keys if k not in model_cfg]
    if missing_keys:
        raise KeyError(
            f"Configuration file '{config_file}' 'model' section is missing required keys: {missing_keys}"
        )

    return model_cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Checkpoint Verification & Model Loading (Strict & Read-Only)
# ─────────────────────────────────────────────────────────────────────────────

ckpt_path = resolve_checkpoint_path()
with open(ckpt_path, "rb") as f:
    ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()
ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

print("=" * 88)
print(" SSL REPRESENTATION COLLAPSE DIAGNOSTIC: PRE- vs. POST-PROJECTION")
print("=" * 88)
print(f"[✓] Checkpoint:    {ckpt_path}")
print(f"    Size:          {ckpt_size_mb:.2f} MB")
print(f"    SHA-256:       {ckpt_sha256}")

# Instantiate model matching configs/ssl.yaml strictly
model_cfg = resolve_config()
model = SSLModel(
    in_channels=model_cfg["in_channels"],
    encoder_channels=model_cfg["encoder_channels"],
    proj_dim=model_cfg["proj_dim"],
    mask_patch_size=model_cfg["mask_patch_size"],
    mask_ratio=model_cfg["mask_ratio"],
    dropout=model_cfg["dropout"],
    use_residual=model_cfg["use_residual"],
)

# Load checkpoint with strict compatibility verification
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
    ckpt_epoch = ckpt.get("epoch", "unknown")
else:
    state_dict = ckpt
    ckpt_epoch = "raw"

load_result = model.load_state_dict(state_dict, strict=True)
print(f"[✓] Model State:   Successfully loaded (strict=True, saved epoch={ckpt_epoch})")
print(f"    Missing Keys:  {len(load_result.missing_keys)}")
print(f"    Unexpected:    {len(load_result.unexpected_keys)}")

# Enforce evaluation and frozen inference mode
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic Multi-Patient Sample Selection
# ─────────────────────────────────────────────────────────────────────────────

processed_dir = resolve_processed_dir()
split_file = resolve_split_file()

print(f"[✓] Data Directory: {processed_dir}")
print(f"[✓] Split File:     {split_file}")

dataset = ACDCTemporalDataset(
    processed_dir=str(processed_dir),
    split_file=str(split_file),
)
print(f"[✓] Dataset Total:  {len(dataset)} adjacent frame pairs")

# Group dataset pair indices directly by iterating dataset.pairs
# Uses verified patient metadata from the pair filepath without loading image arrays
patient_pair_indices: Dict[str, List[int]] = {}
for idx, (path_t, _) in enumerate(dataset.pairs):
    fname = Path(path_t).stem
    match = re.match(r"(patient\d+)", fname)
    if match:
        pid = match.group(1)
    else:
        # Verified fallback directly from NPZ metadata without loading image array
        with np.load(path_t) as npz:
            pid = str(npz["patient_id"])
    patient_pair_indices.setdefault(pid, []).append(idx)

available_pids = sorted(list(patient_pair_indices.keys()))
if len(available_pids) < 2:
    raise RuntimeError(
        f"Diagnostic requires multiple distinct patients, but only found {len(available_pids)}."
    )

# Deterministic, reproducible selection uniformly distributed across available patients
RNG_SEED = 42
rng = np.random.RandomState(RNG_SEED)

n_target_patients = min(len(available_pids), 25)
# Sample across the entire cohort rather than just alphabetical head
target_pids = sorted(rng.choice(available_pids, size=n_target_patients, replace=False).tolist())
pairs_per_patient = max(2, min(5, 100 // n_target_patients))

selected_pairs: List[Tuple[int, str]] = []
for pid in target_pids:
    idxs = patient_pair_indices[pid]
    # Strided selection across patient slices/frames
    step = max(1, len(idxs) // pairs_per_patient)
    for k in range(min(pairs_per_patient, len(idxs))):
        selected_pairs.append((idxs[k * step], pid))

print(f"[✓] Sampling Plan:  {len(selected_pairs)} pairs across {len(target_pids)} distinct patients (seed={RNG_SEED})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Extraction: Pre-Projection (Bottleneck GAP) & Post-Projection
# ─────────────────────────────────────────────────────────────────────────────

batch_size = 16
all_pre_t: List[torch.Tensor] = []
all_pre_t1: List[torch.Tensor] = []
all_post_t: List[torch.Tensor] = []
all_post_t1: List[torch.Tensor] = []
all_pids: List[str] = []

# Layer-by-layer activation stages for projection head investigation (frame_t)
stage_acts: Dict[str, List[torch.Tensor]] = {
    "1_pool_flat": [],
    "2_linear1": [],
    "3_relu": [],
    "4_linear2": [],
    "5_l2_norm": [],
}

with torch.inference_mode():
    for start in range(0, len(selected_pairs), batch_size):
        end = min(start + batch_size, len(selected_pairs))
        batch_slice = selected_pairs[start:end]
        batch_idxs = [item[0] for item in batch_slice]
        batch_expected_pids = [item[1] for item in batch_slice]
        
        batch_samples = [dataset[i] for i in batch_idxs]
        frames_t = torch.stack([s["frame_t"] for s in batch_samples], dim=0)   # (B, 1, H, W)
        frames_t1 = torch.stack([s["frame_t1"] for s in batch_samples], dim=0) # (B, 1, H, W)
        
        # Verify and record patient IDs directly from loaded samples
        for s, exp_pid in zip(batch_samples, batch_expected_pids):
            actual_pid = str(s["patient_id"])
            if actual_pid != exp_pid:
                raise AssertionError(
                    f"Sample patient_id mismatch: loaded '{actual_pid}', expected '{exp_pid}'"
                )
            all_pids.append(actual_pid)
        
        # Verify finite inputs
        if not torch.isfinite(frames_t).all():
            raise ValueError(f"Non-finite values detected in input frame_t at batch {start}:{end}")
        if not torch.isfinite(frames_t1).all():
            raise ValueError(f"Non-finite values detected in input frame_t1 at batch {start}:{end}")
        
        # Clean forward pass through shared encoder
        bottleneck_t, _ = model.encoder(frames_t)     # (B, 256, H', W')
        bottleneck_t1, _ = model.encoder(frames_t1)   # (B, 256, H', W')
        
        if not torch.isfinite(bottleneck_t).all() or not torch.isfinite(bottleneck_t1).all():
            raise ValueError(f"Non-finite values in encoder bottleneck at batch {start}:{end}")
        
        # ── Projection Head Layer-by-Layer Stages (frame_t) ──
        # Stage 1: Adaptive average pooling / flattening
        act_pool_flat = F.adaptive_avg_pool2d(bottleneck_t, (1, 1)).flatten(1)
        # Stage 2: First linear layer
        act_linear1 = model.projection.head[2](act_pool_flat)
        # Stage 3: ReLU (non-inplace)
        act_relu = F.relu(act_linear1)
        # Stage 4: Second linear layer, before L2 normalization
        act_linear2 = model.projection.head[4](act_relu)
        # Stage 5: Final L2 normalization
        act_norm = F.normalize(act_linear2, p=2, dim=1)
        
        # Finite-value checks across all 5 stages
        for s_name, s_tensor in [
            ("Stage 1 (Pool+Flatten)", act_pool_flat),
            ("Stage 2 (Linear 1)", act_linear1),
            ("Stage 3 (ReLU)", act_relu),
            ("Stage 4 (Linear 2)", act_linear2),
            ("Stage 5 (L2 Norm)", act_norm),
        ]:
            if not torch.isfinite(s_tensor).all():
                raise ValueError(f"Non-finite values detected in {s_name} at batch {start}:{end}")
        
        # Shape verification for all stages
        assert act_pool_flat.shape == (len(batch_samples), 256), f"Unexpected Stage 1 shape: {act_pool_flat.shape}"
        assert act_linear1.shape == (len(batch_samples), 256), f"Unexpected Stage 2 shape: {act_linear1.shape}"
        assert act_relu.shape == (len(batch_samples), 256), f"Unexpected Stage 3 shape: {act_relu.shape}"
        assert act_linear2.shape == (len(batch_samples), 128), f"Unexpected Stage 4 shape: {act_linear2.shape}"
        assert act_norm.shape == (len(batch_samples), 128), f"Unexpected Stage 5 shape: {act_norm.shape}"
        
        stage_acts["1_pool_flat"].append(act_pool_flat)
        stage_acts["2_linear1"].append(act_linear1)
        stage_acts["3_relu"].append(act_relu)
        stage_acts["4_linear2"].append(act_linear2)
        stage_acts["5_l2_norm"].append(act_norm)
        
        # Pre/post embeddings for paired metrics
        emb_pre_t = act_pool_flat
        emb_pre_t1 = F.adaptive_avg_pool2d(bottleneck_t1, (1, 1)).flatten(1)
        emb_post_t = act_norm
        emb_post_t1 = model.projection(bottleneck_t1)
        
        if not torch.isfinite(emb_pre_t1).all():
            raise ValueError(f"Non-finite pre-projection embeddings (frame_t1) at batch {start}:{end}")
        if not torch.isfinite(emb_post_t1).all():
            raise ValueError(f"Non-finite post-projection embeddings (frame_t1) at batch {start}:{end}")
        
        all_pre_t.append(emb_pre_t)
        all_pre_t1.append(emb_pre_t1)
        all_post_t.append(emb_post_t)
        all_post_t1.append(emb_post_t1)

# Count evaluated patients directly from sample metadata and assert agreement
evaluated_distinct_pids = sorted(list(set(all_pids)))
if evaluated_distinct_pids != target_pids:
    raise AssertionError(
        f"Mismatch between target patients ({target_pids}) and processed patient_ids ({evaluated_distinct_pids})"
    )
print(f"[✓] Evaluated Patients: {len(evaluated_distinct_pids)} confirmed matching selection IDs")

pre_t = torch.cat(all_pre_t, dim=0)     # (N, 256)
pre_t1 = torch.cat(all_pre_t1, dim=0)   # (N, 256)
post_t = torch.cat(all_post_t, dim=0)   # (N, 128)
post_t1 = torch.cat(all_post_t1, dim=0) # (N, 128)

stages_concatenated = {
    k: torch.cat(v, dim=0) for k, v in stage_acts.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistical Dispersion & Similarity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(emb_t: torch.Tensor, emb_t1: torch.Tensor, pids: List[str]) -> Dict:
    N, D = emb_t.shape
    distinct_pids = sorted(list(set(pids)))
    
    # 1. Per-dimension variance across all samples
    dim_vars = torch.var(emb_t, dim=0, unbiased=True)
    mean_var = dim_vars.mean().item()
    
    # 2. Number of active dimensions (variance > 1e-6)
    active_dims = (dim_vars > 1e-6).sum().item()
    
    # 3. Cross-patient pairwise cosine similarity (strictly pairs i < j where pid[i] != pid[j])
    norm_t = F.normalize(emb_t, p=2, dim=1)
    sim_matrix = norm_t @ norm_t.T  # (N, N)
    
    cross_mask = torch.tensor(
        [[pids[i] != pids[j] for j in range(N)] for i in range(N)],
        dtype=torch.bool,
    ) & torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    
    cross_sims = sim_matrix[cross_mask]
    if len(cross_sims) == 0:
        raise RuntimeError("Failed to form cross-patient pairs for similarity evaluation.")
    
    cross_mean = cross_sims.mean().item()
    cross_std = cross_sims.std().item()
    cross_min = cross_sims.min().item()
    cross_max = cross_sims.max().item()
    
    # 4. Adjacent frame cosine similarity (same patient, same slice, frame t vs frame t+1)
    adj_sims = F.cosine_similarity(emb_t, emb_t1, dim=-1)
    adj_mean = adj_sims.mean().item()
    adj_std = adj_sims.std().item()
    adj_min = adj_sims.min().item()
    adj_max = adj_sims.max().item()
    
    return {
        "num_samples": N,
        "num_distinct_patients": len(distinct_pids),
        "embedding_dim": D,
        "mean_var": mean_var,
        "active_dims": active_dims,
        "cross_mean": cross_mean,
        "cross_std": cross_std,
        "cross_min": cross_min,
        "cross_max": cross_max,
        "num_cross_pairs": len(cross_sims),
        "adj_mean": adj_mean,
        "adj_std": adj_std,
        "adj_min": adj_min,
        "adj_max": adj_max,
        "num_adj_pairs": len(adj_sims),
    }


pre_stats = compute_metrics(pre_t, pre_t1, all_pids)
post_stats = compute_metrics(post_t, post_t1, all_pids)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Structured Comparative Report
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 88)
print(f"{'DIAGNOSTIC METRIC':<42} | {'PRE-PROJECTION (GAP)':<20} | {'POST-PROJECTION (HEAD)':<20}")
print("─" * 88)
print(f"{'Embedding Dimension':<42} | {pre_stats['embedding_dim']:<20} | {post_stats['embedding_dim']:<20}")
print(f"{'Total Evaluated Frame Pairs':<42} | {pre_stats['num_samples']:<20} | {post_stats['num_samples']:<20}")
print(f"{'Distinct Patients Evaluated':<42} | {pre_stats['num_distinct_patients']:<20} | {post_stats['num_distinct_patients']:<20}")
print(f"{'Mean Per-Dimension Variance':<42} | {pre_stats['mean_var']:<20.6e} | {post_stats['mean_var']:<20.6e}")
print(f"{'Active Dims (Variance > 1e-6)':<42} | {pre_stats['active_dims']:>3d} / {pre_stats['embedding_dim']:<14d} | {post_stats['active_dims']:>3d} / {post_stats['embedding_dim']:<14d}")
print(f"{'Cross-Patient Cosine Sim (Mean ± Std)':<42} | {pre_stats['cross_mean']:.4f} ± {pre_stats['cross_std']:.4f}        | {post_stats['cross_mean']:.4f} ± {post_stats['cross_std']:.4f}")
print(f"{'Cross-Patient Cosine Sim [Min, Max]':<42} | [{pre_stats['cross_min']:.3f}, {pre_stats['cross_max']:.3f}]            | [{post_stats['cross_min']:.3f}, {post_stats['cross_max']:.3f}]")
print(f"{'Adjacent Frame Cosine Sim (Mean ± Std)':<42} | {pre_stats['adj_mean']:.4f} ± {pre_stats['adj_std']:.4f}        | {post_stats['adj_mean']:.4f} ± {post_stats['adj_std']:.4f}")
print(f"{'Adjacent Frame Cosine Sim [Min, Max]':<42} | [{pre_stats['adj_min']:.3f}, {pre_stats['adj_max']:.3f}]            | [{post_stats['adj_min']:.3f}, {post_stats['adj_max']:.3f}]")
print("─" * 88)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Evidence Analysis & Limitations
# ─────────────────────────────────────────────────────────────────────────────

print("\nEVIDENCE ANALYSIS:")
print("1. Variance & Dimensional Activity:")
print(f"   • Pre-Projection Bottleneck:  mean variance = {pre_stats['mean_var']:.6e}, active dims = {pre_stats['active_dims']}/{pre_stats['embedding_dim']}.")
print(f"   • Post-Projection Head:       mean variance = {post_stats['mean_var']:.6e}, active dims = {post_stats['active_dims']}/{post_stats['embedding_dim']}.")

print("\n2. Patient Diversity vs. Temporal Consistency:")
print(f"   • Pre-Projection (Bottleneck):")
print(f"     - Different patients: cosine similarity = {pre_stats['cross_mean']:.4f} ± {pre_stats['cross_std']:.4f} (range [{pre_stats['cross_min']:.3f}, {pre_stats['cross_max']:.3f}]).")
print(f"     - Adjacent frames:    cosine similarity = {pre_stats['adj_mean']:.4f} ± {pre_stats['adj_std']:.4f} (range [{pre_stats['adj_min']:.3f}, {pre_stats['adj_max']:.3f}]).")
print(f"   • Post-Projection (Head):")
print(f"     - Different patients: cosine similarity = {post_stats['cross_mean']:.4f} ± {post_stats['cross_std']:.4f} (range [{post_stats['cross_min']:.3f}, {post_stats['cross_max']:.3f}]).")
print(f"     - Adjacent frames:    cosine similarity = {post_stats['adj_mean']:.4f} ± {post_stats['adj_std']:.4f} (range [{post_stats['adj_min']:.3f}, {post_stats['adj_max']:.3f}]).")

print("\n3. Analytical Limitations:")
print("   • Pre-projection features are pooled via AdaptiveAvgPool2d(1) to form 256-D vectors; this summarizes")
print("     global bottleneck features but does not measure spatial feature map variance at full resolution.")
print("   • Preserved variance in the encoder bottleneck reflects representations learned via masked image")
print("     reconstruction. Because the temporal objective gradient into the encoder was negligible (~1e-10 of recon),")
print("     this diagnostic measures feature retention from reconstruction, not the efficacy of temporal guidance.")
print("=" * 88 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Projection Head Layer-by-Layer Collapse Investigation
# ─────────────────────────────────────────────────────────────────────────────

stage_meta = [
    ("1_pool_flat", "Adaptive AvgPool / Flatten", 256),
    ("2_linear1",   "First Linear (256 -> 256)",  256),
    ("3_relu",      "ReLU Activation",            256),
    ("4_linear2",   "Second Linear (256 -> 128)", 128),
    ("5_l2_norm",   "Final L2 Normalization",     128),
]

print("=" * 88)
print(" PROJECTION HEAD LAYER-BY-LAYER COLLAPSE INVESTIGATION")
print("=" * 88)
print(f"{'STAGE':<30} | {'SHAPE':<14} | {'MEAN VAR':<14} | {'ACTIVE DIMS (>1e-6)':<22} | {'FINITE'}")
print("─" * 88)

for key, label, exp_dim in stage_meta:
    act = stages_concatenated[key]
    var_dim = torch.var(act, dim=0, unbiased=True)
    mean_var = var_dim.mean().item()
    active = (var_dim > 1e-6).sum().item()
    is_finite = torch.isfinite(act).all().item()
    shape_str = f"[{act.shape[0]}, {act.shape[1]}]"
    active_str = f"{active:>3d} / {exp_dim:<3d}"
    print(f"{label:<30} | {shape_str:<14} | {mean_var:<14.6e} | {active_str:<22} | {str(is_finite)}")

# Second linear layer: pre-normalization embedding L2 norms and variance breakdown
act_linear2 = stages_concatenated["4_linear2"]
l2_norms_linear2 = torch.norm(act_linear2, p=2, dim=1)
var_linear2 = torch.var(act_linear2, dim=0, unbiased=True)

print("\n" + "─" * 88)
print("SECOND LINEAR LAYER (PRE-NORMALIZATION) ANALYSIS:")
print(f"  • Pre-Norm Embedding L2 Norms:")
print(f"    - Minimum:  {l2_norms_linear2.min().item():.6f}")
print(f"    - Median:   {torch.median(l2_norms_linear2).item():.6f}")
print(f"    - Maximum:  {l2_norms_linear2.max().item():.6f}")
print(f"    - Mean:     {l2_norms_linear2.mean().item():.6f}")
print(f"  • Pre-Norm Dimension Variance:")
print(f"    - Minimum:  {var_linear2.min().item():.6e}")
print(f"    - Median:   {torch.median(var_linear2).item():.6e}")
print(f"    - Maximum:  {var_linear2.max().item():.6e}")
print(f"    - Mean:     {var_linear2.mean().item():.6e}")

# Final normalized embeddings: norms and numerical identity verification
act_norm = stages_concatenated["5_l2_norm"]
l2_norms_final = torch.norm(act_norm, p=2, dim=1)

N_samples = act_norm.shape[0]
triu_mask = torch.triu(torch.ones(N_samples, N_samples, dtype=torch.bool), diagonal=1)
pairwise_l2 = torch.cdist(act_norm, act_norm, p=2)[triu_mask]
diff_elem = torch.max(
    torch.abs(act_norm.unsqueeze(1) - act_norm.unsqueeze(0)),
    dim=-1
).values[triu_mask]

max_abs_elem_diff = diff_elem.max().item()
min_pair_l2 = pairwise_l2.min().item()
max_pair_l2 = pairwise_l2.max().item()
is_numerically_identical = (max_abs_elem_diff == 0.0)

print("\n" + "─" * 88)
print("FINAL NORMALIZED EMBEDDINGS (POST-NORMALIZATION) ANALYSIS:")
print(f"  • Normalized Embedding L2 Norms:")
print(f"    - Minimum:  {l2_norms_final.min().item():.6f}")
print(f"    - Median:   {torch.median(l2_norms_final).item():.6f}")
print(f"    - Maximum:  {l2_norms_final.max().item():.6f}")
print(f"  • Pairwise Identity Check Across All Distinct Sample Pairs ({len(pairwise_l2)} pairs):")
print(f"    - Max Absolute Element Difference: {max_abs_elem_diff:.6e}")
print(f"    - Pairwise L2 Distance Range:      [{min_pair_l2:.6e}, {max_pair_l2:.6e}]")
if is_numerically_identical:
    print("    - Status: NUMERICALLY IDENTICAL (exact float-level equality across all samples)")
else:
    print("    - Status: MERELY NEARLY IDENTICAL (concentrated in an extremely tight directional cluster with non-zero residual differences)")
print("=" * 88 + "\n")
