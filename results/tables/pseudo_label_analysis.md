# Table D: Pseudo-Label Analysis

Analysis of pseudo-label generation across 15,860 unlabeled intermediate cine slices (100% training cohort, Kaggle run) and downstream test evaluation on the canonical 20-patient ACDC test cohort (398 slices, seed 42).

| Method / Filtering Configuration | Pseudo Slices | Acceptance Rate (%) | Rejection Rate (%) | Min Confidence ($p_{max}$) | Temporal Consist. | Downstream Mean Dice | HD95 (mm) | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **No Pseudo-Labels (SSL + Motion)** | 0 | — | — | — | — | **0.7723** | **10.34** | Supervised ED/ES + motion loss only (no pseudo-labels) |
| **Raw / Unfiltered Pseudo-Labels (+ Motion)** | 15,860 | 100.0% | 0.0% | 0.00 | No | **0.7868** | **9.19** | Direct teacher argmax without filtering (full_pipeline_raw) |
| **Confidence-Filtered Full Pipeline (+ Motion)** | 15,860 | 98.50% | 1.50% | 0.90 | Yes | **0.7905** | **13.26** | MSP x temporal agreement >= 0.90 + motion regularization |
| **Confidence-Filtered SSL + Pseudo (No Motion)** | 15,860 | 98.50% | 1.50% | 0.90 | Yes | **0.7941** | **7.17** | Filtered 100% pseudo-label pool without fine-tuning motion reg. (Variant D) |

### Filtering Rules & Technical Specifications
- **Active 100% Pool Generation (`configs/pseudo_labels.yaml`)**: `confidence_metric: max_probability` ($p_{max} = \max_c P(y=c|x) \ge 0.90$) modulated by motion-warped temporal agreement (`use_temporal_consistency: true`, `confidence * agreement >= 0.90` in `src/pseudo_labels.py`).
- **100% Pool Acceptance Statistics (Kaggle)**: Generated **15,860** unlabeled intermediate cine pseudo-labels (`results/pseudo_labels/100pct/pseudo_label_metadata.json`) with **98.50%** pixel acceptance (**1.50%** pixel rejection).
- **Configured Entropy Parameter**: `entropy_threshold: 0.25` in `configs/pseudo_labels.yaml` (available for `entropy`/`hybrid` modes).
- **Minimum Foreground Area**: Slices with fewer than `30` accepted foreground pixels are rejected (`ignore_index = -1`).

### Data Provenance & Integrity Checksums
- 100% Kaggle pseudo-label pool: `results/pseudo_labels/100pct/pseudo_label_metadata.json` (`15,860` slices, `98.50%` pixel acceptance; note `results/pseudo_labels/pseudo_label_metadata.json` SHA-256 `66f04eec2130b4e9...` records an earlier 2-sample local validation check at `0.47%` acceptance)
- Pseudo-label generator module: `src/pseudo_labels.py` (SHA-256: `205a8c7f07b49911...`)
- Pseudo-label config: `configs/pseudo_labels.yaml` (SHA-256: `7b34e8f6f2e2cb41...`)
- No Pseudo (SSL + Motion): `results/experiments/ssl_motion_100pct_seed42/test_metrics.json` (SHA-256: `f1eeabf9a6ddc34b...`)
- Raw / Unfiltered (+ Motion): `results/experiments/full_pipeline_raw_100pct_seed42/test_metrics.json` (SHA-256: `210a0490e92c9821...`)
- Filtered Full Pipeline: `results/experiments/full_pipeline_100pct_seed42/test_metrics.json` (SHA-256: `6391c06dcd2888d2...`)
- Filtered SSL + Pseudo (No Motion): `results/experiments/ssl_pseudo_100pct_seed42/test_metrics.json` (SHA-256: `3e44cf82c42ca32c...`)
