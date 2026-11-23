# Table A: Main Segmentation Results

Evaluation on the canonical 20-patient ACDC test cohort (398 slices, seed 42, $256 \times 256$ in-plane resolution, spacing $(1.5, 1.5)\text{ mm}$).

| Method | LV Dice | RV Dice | MYO Dice | Mean Dice | HD95 (mm) | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TransUNet (Full)** | 0.8776 | 0.7637 | 0.8152 | **0.8189** | **3.70** | Hybrid CNN-ViT SOTA baseline |
| **Supervised U-Net (100% Baseline)** | 0.8644 | 0.7554 | 0.7992 | **0.8064** | **6.23** | Canonical MONAI 2D U-Net baseline |
| **SegFormer (Full)** | 0.8680 | 0.7558 | 0.7878 | **0.8038** | **3.77** | MiT-B2 hierarchical transformer |
| **Swin-UNet (Full)** | 0.8789 | 0.7133 | 0.8061 | **0.7994** | **4.25** | Pure Swin-Tiny transformer U-Net |
| **Full Pipeline (Ours, 100%)** | 0.8533 | 0.7244 | 0.7938 | **0.7905** | **13.26** | Proposed motion-guided SSL framework |
| **MedSAM (Fine-Tuned Decoder, 100%)** | 0.6090 | 0.4090 | 0.4511 | **0.4897** | **20.94** | Foundation model (ViT-B) fine-tuned decoder |
| **MedSAM (Zero-Shot)** | 0.1751 | 0.1127 | 0.2492 | **0.1790** | **37.64** | Zero-shot inference with training priors |

### Data Provenance & Integrity Checksums
- TransUNet test metrics: `results/experiments/baseline_transunet_full/test_metrics.json` (SHA-256: `386d89603800d83c...`)
- SegFormer test metrics: `results/experiments/baseline_segformer_full_20261018/test_metrics.json` (SHA-256: `9735413977b381c9...`)
- Swin-UNet test metrics: `results/experiments/baseline_swin_unet/test_metrics.json` (SHA-256: `df48bf3453c3fa54...`)
- Full Pipeline 100% test metrics: `results/experiments/full_pipeline_100pct_seed42/test_metrics.json` (SHA-256: `6391c06dcd2888d2...`)
- MedSAM matched comparison: `results/tables/medsam_matched_comparison.md`
