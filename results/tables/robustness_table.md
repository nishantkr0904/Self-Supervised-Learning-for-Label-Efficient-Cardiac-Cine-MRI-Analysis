# Test-Time Perturbation Robustness Analysis (Step 12)

Evaluated on canonical test cohort (`20` patients, `398` slices, spacing `(1.5, 1.5)`, seed `42`).

| Perturbation / Condition | Target Variant | Mean Dice | LV Dice | MYO Dice | RV Dice | HD95 (mm) | Delta Mean Dice | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Standard (Clean) | Supervised Baseline (100%) | 0.7692 +/- 0.0888 | 0.8491 | 0.7799 | 0.6788 | 21.98 +/- 16.96 | +0.0000 | Evaluated |
| Gaussian Noise (std=0.05) | Supervised Baseline (100%) | 0.7692 +/- 0.0919 | 0.8479 | 0.7782 | 0.6816 | 20.32 +/- 18.42 | -0.0000 | Evaluated |
| Gaussian Noise (std=0.10) | Supervised Baseline (100%) | 0.7516 +/- 0.0948 | 0.8346 | 0.7588 | 0.6615 | 21.91 +/- 19.66 | -0.0176 | Evaluated |
| Intensity Scale (0.9x) | Supervised Baseline (100%) | 0.7708 +/- 0.0885 | 0.8470 | 0.7808 | 0.6844 | 20.75 +/- 15.90 | +0.0015 | Evaluated |
| Intensity Scale (1.1x) | Supervised Baseline (100%) | 0.7685 +/- 0.0907 | 0.8487 | 0.7790 | 0.6777 | 22.87 +/- 18.01 | -0.0008 | Evaluated |
| Standard (Clean) | Full Pipeline (Ours, 100%) | 0.7905 +/- 0.0935 | 0.8533 | 0.7938 | 0.7244 | 13.26 +/- 12.78 | +0.0000 | Evaluated |
| Gaussian Noise (std=0.05) | Full Pipeline (Ours, 100%) | 0.7814 +/- 0.1037 | 0.8426 | 0.7820 | 0.7196 | 16.60 +/- 21.07 | -0.0091 | Evaluated |
| Gaussian Noise (std=0.10) | Full Pipeline (Ours, 100%) | 0.7785 +/- 0.0943 | 0.8497 | 0.7731 | 0.7128 | 13.73 +/- 14.86 | -0.0120 | Evaluated |
| Intensity Scale (0.9x) | Full Pipeline (Ours, 100%) | 0.7915 +/- 0.0968 | 0.8483 | 0.7916 | 0.7344 | 12.47 +/- 12.71 | +0.0010 | Evaluated |
| Intensity Scale (1.1x) | Full Pipeline (Ours, 100%) | 0.7885 +/- 0.0931 | 0.8549 | 0.7935 | 0.7172 | 13.94 +/- 13.62 | -0.0020 | Evaluated |
