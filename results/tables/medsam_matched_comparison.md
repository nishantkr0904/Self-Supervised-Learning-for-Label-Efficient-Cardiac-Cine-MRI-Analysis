# Matched MedSAM Test Comparison

## Protocol

- Test cohort: 20 patients and 398 labeled slices.
- Both methods use the same official MedSAM ViT-B checkpoint and the same 10%-training-derived class-specific prior boxes.
- The zero-shot method has no ACDC decoder fine-tuning.
- The fine-tuned method uses the decoder checkpoint from the 10% pilot, trained on 7 patients; the selected checkpoint is from epoch 15.
- Evaluation uses the canonical test split and patient-level aggregation: slice metrics are averaged within each patient, followed by an unweighted mean and standard deviation across patients.
- Dice is better when higher; HD95 is better when lower.
- MedPy was unavailable, so the fallback HD95 implementation was used.

## Results

| Method | LV Dice | Myocardium Dice | RV Dice | Mean Dice | LV HD95 (mm) | Myocardium HD95 (mm) | RV HD95 (mm) | Mean HD95 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedSAM zero-shot, 10% prior boxes | 0.1374 ± 0.1219 | 0.2282 ± 0.1067 | 0.1258 ± 0.0953 | 0.1638 ± 0.0861 | 32.0897 ± 14.6128 | 35.3514 ± 12.9634 | 55.8831 ± 22.4716 | 40.2840 ± 13.8679 |
| MedSAM fine-tuned decoder, 10% pilot | 0.4584 ± 0.2228 | 0.3164 ± 0.1574 | 0.2398 ± 0.1934 | 0.3382 ± 0.1733 | 23.8891 ± 14.0950 | 27.1812 ± 14.0160 | 36.6993 ± 25.3421 | 29.1067 ± 16.5446 |

## Interpretation

On this test cohort, fine-tuning increased mean Dice by 0.1744 absolute (approximately 106.5% relative to the zero-shot mean) and reduced mean HD95 by 11.1773 mm (approximately 27.7%). These are descriptive comparisons, not evidence of statistical significance. Per-patient variability remains substantial, particularly for RV segmentation.

## Source artifacts

- Zero-shot report: `results/experiments/medsam_finetune_10pct_test_eval_20261023/medsam_10pct_prior_zeroshot_test_metrics.json`
- Fine-tuned report: `results/experiments/medsam_finetune_10pct_test_eval_20261023/medsam_10pct_prior_finetuned_test_metrics.json`
- Prior boxes: `configs/medsam_prior_boxes_10pct.json`
- Fine-tuned decoder: `results/experiments/medsam_finetune_10pct_pilot_20261022/checkpoints/best_decoder.pt`

This matched comparison is separate from the historical full-prior zero-shot baseline in `results/baselines/medsam_zero_shot/summary.csv`; the latter uses a different prompting configuration and should not be substituted for the matched zero-shot row.
