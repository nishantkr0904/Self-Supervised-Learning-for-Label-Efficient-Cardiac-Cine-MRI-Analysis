# Table B: Label-Efficiency Results

Performance across varying proportions of labeled training patients (10%, 25%, 50%, and 100% of 70 training patients).

| Method | 10% Labels | 25% Labels | 50% Labels | 100% Labels | HD95 @ 100% (mm) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Pipeline (Ours)** | 0.6786 | 0.7316 | 0.7547 | **0.7905** | 13.26 mm | Proposed motion-guided SSL |
| **Supervised U-Net (Baseline)** | 0.5807 | 0.6608 | 0.7293 | **0.8064** | 6.23 mm | Standard supervised 2D U-Net |
| **MedSAM (Fine-Tuned)** | 0.3382 | 0.3838 | 0.4398 | **0.4897** | 20.94 mm | MedSAM ViT-B fine-tuned decoder |
| **TransUNet (Full)** | — | — | — | **0.8189** | 3.70 mm | Evaluated at 100% labels only |
| **SegFormer (Full)** | — | — | — | **0.8038** | 3.77 mm | Evaluated at 100% labels only |
| **Swin-UNet (Full)** | — | — | — | **0.7994** | 4.25 mm | Evaluated at 100% labels only |

### Full Pipeline (Ours): Detailed Per-Class Breakdown Across Label Fractions

| Label Fraction | LV Dice | MYO Dice | RV Dice | Mean Dice | HD95 (mm) | Train Cohort | Test Cohort |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 0.7883 | 0.6814 | 0.5661 | **0.6786** | 32.52 | 7 patients (140 slices) | 20 patients (398 slices) |
| **25%** | 0.8217 | 0.7367 | 0.6364 | **0.7316** | 22.24 | 17 patients (340 slices) | 20 patients (398 slices) |
| **50%** | 0.8394 | 0.7664 | 0.6584 | **0.7547** | 17.62 | 35 patients (700 slices) | 20 patients (398 slices) |
| **100%** | 0.8533 | 0.7938 | 0.7244 | **0.7905** | 13.26 | 70 patients (1392 slices) | 20 patients (398 slices) |

### Data Provenance & Integrity Checksums
- Full Pipeline 10%: `results/experiments/full_pipeline_10pct_seed42/test_metrics.json` (SHA-256: `a3786267540bb37f...`)
- Full Pipeline 25%: `results/experiments/full_pipeline_25pct_seed42/test_metrics.json` (SHA-256: `a079539ff6eab185...`)
- Full Pipeline 50%: `results/experiments/full_pipeline_50pct_seed42/test_metrics.json` (SHA-256: `12566ff9f65db6f4...`)
- Full Pipeline 100%: `results/experiments/full_pipeline_100pct_seed42/test_metrics.json` (SHA-256: `6391c06dcd2888d2...`)
- Supervised U-Net 10%: `results/experiments/supervised_10pct_seed42/test_metrics.json` (HD95: `50.78 mm`, SHA-256: `e2ab30e809b926ae...`)
- Supervised U-Net 25%: `results/experiments/supervised_25pct_seed42/test_metrics.json` (HD95: `35.50 mm`, SHA-256: `3c051c22191ba59e...`)
- Supervised U-Net 50%: `results/experiments/supervised_50pct_seed42/test_metrics.json` (HD95: `22.74 mm`, SHA-256: `8b69e54d05ac4c4c...`)
