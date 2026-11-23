# Evidence Audit Report: Motion-Guided Self-Supervised Learning for Label-Efficient Cardiac Cine MRI Analysis

**Date:** 2026-11-20  
**Project Directory:** `/Users/nishant/Documents/motion_guided_ssl_project`  
**Auditor:** Senior Research Auditor (Antigravity Assistant)  
**Primary Source Document:** `Team_02 Motion-Guided Self-Supervised Learning for Label-Efficient Cardiac Cine.pdf` (ACDC Benchmark Specification)  

---

## Section A: Research Objectives (from Specification PDF)

1. **Central Hypothesis & Research Question:**
   - Cardiac cine MRI acquisitions capture continuous cardiac dynamics (contraction and relaxation phases). Most existing segmentation networks process 2D slices independently as static frames, discarding rich temporal correspondence.
   - At the same time, manual expert pixel annotations are prohibitively expensive and time-consuming.
   - **Central Question:** Can self-supervised learning that leverages temporal structural evolution and lightweight motion guidance across adjacent cine frames significantly reduce the requirement for manual annotations in multi-class cardiac segmentation?

2. **Core Research Objectives:**
   - Construct a reproducible 2D cardiac cine MRI baseline (MONAI U-Net / ResUNet).
   - Learn temporal representations from consecutive frames or phase pairs without manual masks during the SSL pretraining stage.
   - Formulate a lightweight motion-guided consistency module (optical flow / displacement warping) rather than complex biomechanical models.
   - Generate confidence-filtered pseudo-labels to provide semi-supervised guidance on unlabeled/weakly labeled cardiac phases.
   - Compare full supervision against limited-label fractions (10%, 25%, 50%, 100% labeled patients).
   - Evaluate whether temporal learning provides disproportionate benefit when manual annotations are scarce.
   - Conduct component-wise ablation studies isolating the contribution of SSL, motion estimation, and pseudo-labeling.

3. **Success Criterion (Section 18 of Specification):**
   - The central metric of success is **label efficiency**, not beating every high-capacity transformer at 100% labels.
   - A major scientific contribution is demonstrated if motion-guided SSL achieves strong segmentation performance at 10% or 25% labels that approaches or exceeds supervised baselines trained with substantially more data.

---

## Section B: Dataset & Splits (Verified from Workspace)

1. **Dataset:**
   - **Identity:** Automated Cardiac Diagnosis Challenge (ACDC 2017) cohort.
   - **Total Cohort:** 100 patients across 5 clinical diagnostic groups (Normals, Ischemic/Myocardial Infarction, Dilated Cardiomyopathy, Hypertrophic Cardiomyopathy, Abnormal RV).
   - **Modality:** Short-axis cardiac cine MRI, end-diastolic (ED) and end-systolic (ES) annotated ground truth.
   - **Classes:** 0: Background, 1: Left Ventricle (LV) Endocardium, 2: Myocardium (MYO), 3: Right Ventricle (RV) Cavity.

2. **Patient-Level Data Splits (Strict Anti-Leakage Protocol):**
   - `data/splits/patient_splits.json`:
     - **Train Split:** 70 patients (`patient001`, `patient003`, ..., `patient100`), yielding 1,392 slices.
     - **Validation Split:** 10 patients (`patient002`, `patient016`, `patient029`, `patient031`, `patient042`, `patient060`, `patient066`, `patient079`, `patient083`, `patient098`), yielding 200 slices.
     - **Test Split:** 20 unseen patients (`patient007`, `patient008`, `patient011`, `patient015`, `patient021`, `patient022`, `patient032`, `patient038`, `patient044`, `patient047`, `patient049`, `patient058`, `patient063`, `patient074`, `patient075`, `patient077`, `patient084`, `patient094`, `patient095`, `patient096`), yielding exactly 398 slices.
   - **Leakage Audit:** All splits are patient-level disjoint. Slices or frames from the same patient never appear in multiple partitions.

3. **Label Fractions:**
   - Fixed subsets of the 70 training patients:
     - **10% Labels:** 7 patients (140 labeled slices)
     - **25% Labels:** 17 patients (340 labeled slices)
     - **50% Labels:** 35 patients (700 labeled slices)
     - **100% Labels:** 70 patients (1,392 labeled slices)

---

## Section C: Architecture & Implementation Details (from Source Code)

1. **Shared Encoder (`src/encoder.py`):**
   - 4-stage residual convolutional hierarchy with channels `[32, 64, 128, 256]`.
   - Each stage utilizes residual convolution blocks (`ResConvBlock`) composed of two $3 \times 3$ convolutions, BatchNorm, and LeakyReLU ($\alpha=0.01$).
   - Downsampling is performed via strided convolutions or max-pooling.
   - Total encoder parameter count: ~1.2M parameters.

2. **Segmentation Decoder (`src/segmentation_model.py`):**
   - Symmetric 4-stage U-Net decoder with transposed convolutions (or bilinear upsampling + conv) and lateral skip connections.
   - Output classification head produces 4 logits (background, LV, MYO, RV) at original spatial resolution ($256 \times 256$).

3. **Self-Supervised Pretraining Module (`src/ssl_pretraining.py`):**
   - Implements two complementary self-supervised objectives:
     1. **Masked Image Reconstruction:** Random patch-based masking (masking ratio $r=0.40$), reconstructed via a lightweight decoder head.
     2. **Adjacent-Frame Temporal Consistency:** Feature-level cosine similarity / $L_2$ distance between representations extracted from adjacent cine frames $(t, t+1)$.

4. **Motion-Guided Consistency Module (`src/motion.py`):**
   - **`SimpleFlowNet`:** Lightweight, fully differentiable 2D optical flow / displacement estimator predicting dense displacement fields $\phi_{t \to t+1} \in \mathbb{R}^{2 \times H \times W}$.
   - **`SpatialTransformer`:** Bilinear grid-sampling operator warping features or segmentation probabilities from time $t$ to $t+1$: $\hat{F}_{t+1} = \mathcal{W}(F_t, \phi_{t \to t+1})$.
   - Regularized with photometric consistency and total-variation displacement smoothness $\mathcal{L}_{\text{smooth}}(\phi)$.

5. **Confidence-Filtered Pseudo-Labeling (`src/pseudo_labels.py`):**
   - Estimates pixel-wise uncertainty using Maximum Softmax Probability ($p_{\max} \ge 0.90$) and Normalized Shannon Entropy ($H \le 0.25$).
   - Enforces temporal consistency thresholding ($> 0.80$) with motion-warped adjacent frames.
   - Masks uncertain pixels with `ignore_index = -1` to prevent error propagation into semi-supervised updates.

---

## Section D: Mathematical Loss Formulations (from Code)

1. **Supervised Segmentation Loss (`src/losses.py`):**
   $$\mathcal{L}_{\text{sup}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$$
   $$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C-1} \sum_{c=1}^{C-1} \frac{2 \sum_i p_{i,c} g_{i,c} + \epsilon}{\sum_i p_{i,c}^2 + \sum_i g_{i,c}^2 + \epsilon}$$

2. **Self-Supervised Pretraining Loss (`src/ssl_pretraining.py`, `src/losses.py`):**
   $$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{temporal}} \mathcal{L}_{\text{temporal}} + \lambda_{\text{motion}} \mathcal{L}_{\text{motion}}$$
   - $\mathcal{L}_{\text{recon}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} |I(i) - \hat{I}(i)|$
   - $\mathcal{L}_{\text{temporal}} = 1 - \frac{\langle z_t, z_{t+1} \rangle}{\|z_t\|_2 \|z_{t+1}\|_2}$
   - $\mathcal{L}_{\text{motion}} = \| I_{t+1} - \mathcal{W}(I_t, \phi_{t \to t+1}) \|_1 + \gamma_{\text{smooth}} \| \nabla \phi \|_1$

3. **Confidence-Filtered Pseudo-Supervision Loss (`src/losses.py`):**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \lambda_{\text{pseudo}} \cdot \frac{1}{N_{\text{valid}}} \sum_{i: w_i = 1} \ell_{\text{CE}}(p_i, \hat{y}_i^{\text{pseudo}})$$
   where $w_i = \mathbb{I}[p_{\max}(i) \ge 0.90 \land H(i) \le 0.25]$.

---

## Section E: Training Configurations (from `configs/*.yaml`)

- **Resolution:** In-plane $256 \times 256$, target resampled spacing $1.5 \times 1.5\text{ mm}^2$.
- **Batch Size:** 8 (for pretraining) / 16 (for fine-tuning).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1\times 10^{-4}$).
- **Learning Rate:** Initial $1\times 10^{-4}$, Cosine Annealing learning rate schedule down to $1\times 10^{-6}$.
- **Pretraining Epochs:** 100 epochs.
- **Fine-Tuning Epochs:** 150 epochs with early stopping (patience = 20 epochs on validation Mean Dice).
- **Random Seed:** 42 across all benchmarked runs.

---

## Section F: Complete Experiment Matrix & Evidence Traceability

All metrics verified on canonical 20-patient ACDC test split (398 slices):

| Experiment Key | Model / Method | Seed | Fraction | Mean Dice | LV Dice | MYO Dice | RV Dice | HD95 (mm) | Source Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `baseline_transunet_full` | TransUNet (R50-ViT) | 42 | 100% | 0.8189 | 0.8776 | 0.8152 | 0.7637 | 3.70 | `results/experiments/baseline_transunet_full/test_metrics.json` |
| `unet_100pct_seed42` | Supervised U-Net | 42 | 100% | 0.8064 | 0.8644 | 0.7992 | 0.7554 | 6.23 | `/kaggle/working/unet_100pct_seed42_artifacts.zip` |
| `baseline_segformer_full` | SegFormer (MiT-B2) | 42 | 100% | 0.8038 | 0.8680 | 0.7878 | 0.7558 | 3.77 | `results/experiments/baseline_segformer_full_20261018/test_metrics.json` |
| `baseline_swin_unet` | Swin-UNet (Swin-T) | 42 | 100% | 0.7994 | 0.8789 | 0.8061 | 0.7133 | 4.25 | `results/experiments/baseline_swin_unet/test_metrics.json` |
| `full_pipeline_100pct` | Full Pipeline (Ours) | 42 | 100% | 0.7905 | 0.8533 | 0.7938 | 0.7244 | 13.26 | `results/experiments/full_pipeline_100pct_seed42/test_metrics.json` |
| `full_pipeline_50pct` | Full Pipeline (Ours) | 42 | 50% | 0.7547 | 0.8394 | 0.7664 | 0.6584 | 17.62 | `results/experiments/full_pipeline_50pct_seed42/test_metrics.json` |
| `full_pipeline_25pct` | Full Pipeline (Ours) | 42 | 25% | 0.7316 | 0.8217 | 0.7367 | 0.6364 | 22.24 | `results/experiments/full_pipeline_25pct_seed42/test_metrics.json` |
| `full_pipeline_10pct` | Full Pipeline (Ours) | 42 | 10% | 0.6786 | 0.7883 | 0.6814 | 0.5661 | 32.52 | `results/experiments/full_pipeline_10pct_seed42/test_metrics.json` |
| `ssl_finetune_100pct` | Variant B (SSL Only) | 42 | 100% | 0.7898 | 0.8507 | 0.7838 | 0.7348 | 8.42 | `results/experiments/ssl_finetune_100pct_seed42/test_metrics.json` |
| `ssl_motion_100pct` | Variant C (SSL+Motion) | 42 | 100% | 0.7723 | 0.8441 | 0.7855 | 0.6871 | 10.34 | `results/experiments/ssl_motion_100pct_seed42/test_metrics.json` |
| `ssl_pseudo_100pct` | Variant D (SSL+Pseudo)| 42 | 100% | 0.7941 | 0.8507 | 0.7888 | 0.7428 | 7.17 | `results/experiments/ssl_pseudo_100pct_seed42/test_metrics.json` |
| `medsam_finetuned` | MedSAM Fine-tuned | 42 | 100% | 0.4897 | 0.6090 | 0.4511 | 0.4090 | 20.94 | `results/tables/medsam_matched_comparison.md` |
| `medsam_zeroshot` | MedSAM Zero-shot | 42 | 100% | 0.1790 | 0.1751 | 0.2492 | 0.1127 | 37.64 | `results/baselines/medsam_zero_shot/medsam_zeroshot_metrics.json` |

---

## Section G: Available Figures & Visualizations

1. **Pre-existing Visualizations:**
   - Preprocessing verification plots in `results/preprocessing/`.
   - Calibration TensorBoard logs in `results/calibration/unet_calibration_100pct_seed42/`.
   - Sample prediction masks in `results/figures/`.
2. **Planned Publication Figures to Generate:**
   - **Figure 1:** End-to-end Method Architecture Diagram (TikZ / vector PDF).
   - **Figure 2:** Label-Efficiency Curves across 10%, 25%, 50%, 100% labeled fractions.
   - **Figure 3:** Component Ablation Bar Chart (Variant A through E).
   - **Figure 4:** Baseline Comparative Benchmark (Ours vs. U-Net, Swin-UNet, SegFormer, TransUNet, MedSAM).
   - **Figure 5:** Per-Patient Test Set Dice Distribution across all 20 evaluation subjects.

---

## Section H: Discrepancies & Methodological Boundaries

1. **Label-Efficiency Baselines:**
   - While the proposed Full Pipeline was trained across all 4 label fractions (10%, 25%, 50%, 100%), heavy transformer baselines (TransUNet, Swin-UNet, SegFormer) were evaluated only at 100% labels due to compute constraints.
   - Supervised U-Net baseline was evaluated at 100% on the test set (0.8064), while its 10%, 25%, and 50% training runs are recorded via validation early-stopping checkpoints (0.6056, 0.6932, 0.7737).
2. **HD95 Evaluation Implementation:**
   - Evaluation scripts utilized fallback NumPy Euclidean distance transforms when MedPy was absent, and MedPy surface distance when installed. The HD95 audit confirmed rank consistency.
3. **No Synthetic Extrapolations:**
   - All presented values in the paper must be strictly traceable to the hashes verified in `update_verified_tables.py` and `audit_and_backups/`.
