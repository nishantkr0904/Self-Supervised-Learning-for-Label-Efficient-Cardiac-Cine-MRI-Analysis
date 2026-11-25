# Section 17: Paired Statistical Analysis

Two-sided paired Wilcoxon signed-rank tests (`zero_method="wilcox"`, `alpha=0.05`) over `n=20` matched canonical test patients, with Holm-Bonferroni step-down correction within each comparison family per metric. `Paired Δ` is computed as `A - B` (`Mean_Dice` higher better; `Mean_HD95` negative means lower error).

| Family | Comparison | Metric | Mean A ± Std A | Mean B ± Std B | Paired Δ | W-stat | Raw p | Holm p | Significant (α=0.05) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Label-Efficiency (m=3) | Full Pipeline (10%) vs. Supervised (10%) | `Mean_Dice` | 0.6786 +/- 0.1491 | 0.5807 +/- 0.1780 | +0.0979 | 1.0 | 3.81e-06 | 7.63e-06 | **Yes** |
| Label-Efficiency (m=3) | Full Pipeline (25%) vs. Supervised (25%) | `Mean_Dice` | 0.7316 +/- 0.1256 | 0.6608 +/- 0.1589 | +0.0708 | 0.0 | 1.91e-06 | 5.72e-06 | **Yes** |
| Label-Efficiency (m=3) | Full Pipeline (50%) vs. Supervised (50%) | `Mean_Dice` | 0.7547 +/- 0.1298 | 0.7293 +/- 0.1247 | +0.0254 | 22.0 | 0.0010 | 0.0010 | **Yes** |
| Ablation (m=4) | +Motion vs. +SSL (100%) | `Mean_Dice` | 0.7723 +/- 0.1177 | 0.7898 +/- 0.1043 | -0.0175 | 43.0 | 0.0192 | 0.0769 | No |
| Ablation (m=4) | +Pseudo (Raw) vs. +Motion (100%) | `Mean_Dice` | 0.7868 +/- 0.0978 | 0.7723 +/- 0.1177 | +0.0145 | 46.0 | 0.0266 | 0.0799 | No |
| Ablation (m=4) | +Conf Filter (Full) vs. +Pseudo Raw (100%) | `Mean_Dice` | 0.7905 +/- 0.0935 | 0.7868 +/- 0.0978 | +0.0037 | 81.0 | 0.3884 | 0.4049 | No |
| Ablation (m=4) | Alt. (SSL+Filtered) vs. Full Pipeline (100%) | `Mean_Dice` | 0.7941 +/- 0.1064 | 0.7905 +/- 0.0935 | +0.0036 | 70.0 | 0.2024 | 0.4049 | No |
| Robustness vs. Clean (m=4) | Full Noise (std=0.05) vs. Clean | `Mean_Dice` | 0.7814 +/- 0.1037 | 0.7905 +/- 0.0935 | -0.0091 | 57.0 | 0.0759 | 0.2276 | No |
| Robustness vs. Clean (m=4) | Full Noise (std=0.10) vs. Clean | `Mean_Dice` | 0.7785 +/- 0.0943 | 0.7905 +/- 0.0935 | -0.0120 | 49.0 | 0.0362 | 0.1449 | No |
| Robustness vs. Clean (m=4) | Full Scale (0.9x) vs. Clean | `Mean_Dice` | 0.7915 +/- 0.0968 | 0.7905 +/- 0.0935 | +0.0010 | 104.0 | 0.9854 | 1.0000 | No |
| Robustness vs. Clean (m=4) | Full Scale (1.1x) vs. Clean | `Mean_Dice` | 0.7885 +/- 0.0931 | 0.7905 +/- 0.0935 | -0.0020 | 89.0 | 0.5706 | 1.0000 | No |
| Robustness Full vs. Sup. (m=5) | Clean: Full vs. Sup. (handoff) | `Mean_Dice` | 0.7905 +/- 0.0935 | 0.7692 +/- 0.0888 | +0.0213 | 21.0 | 0.0009 | 0.0034 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Noise (0.05): Full vs. Sup. (handoff) | `Mean_Dice` | 0.7814 +/- 0.1037 | 0.7692 +/- 0.0919 | +0.0122 | 41.0 | 0.0153 | 0.0153 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Noise (0.10): Full vs. Sup. (handoff) | `Mean_Dice` | 0.7785 +/- 0.0943 | 0.7516 +/- 0.0948 | +0.0269 | 25.0 | 0.0017 | 0.0051 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Scale (0.9x): Full vs. Sup. (handoff) | `Mean_Dice` | 0.7915 +/- 0.0968 | 0.7708 +/- 0.0885 | +0.0207 | 25.0 | 0.0017 | 0.0051 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Scale (1.1x): Full vs. Sup. (handoff) | `Mean_Dice` | 0.7885 +/- 0.0931 | 0.7685 +/- 0.0907 | +0.0200 | 15.0 | 0.0003 | 0.0013 | **Yes** |
| Label-Efficiency (m=3) | Full Pipeline (10%) vs. Supervised (10%) | `Mean_HD95` | 32.52 +/- 26.05 | 50.78 +/- 33.59 | -18.26 | 24.0 | 0.0014 | 0.0043 | **Yes** |
| Label-Efficiency (m=3) | Full Pipeline (25%) vs. Supervised (25%) | `Mean_HD95` | 22.24 +/- 20.04 | 35.50 +/- 27.60 | -13.26 | 35.0 | 0.0073 | 0.0146 | **Yes** |
| Label-Efficiency (m=3) | Full Pipeline (50%) vs. Supervised (50%) | `Mean_HD95` | 17.62 +/- 16.83 | 22.74 +/- 18.34 | -5.12 | 47.0 | 0.0296 | 0.0296 | **Yes** |
| Ablation (m=4) | +Motion vs. +SSL (100%) | `Mean_HD95` | 10.34 +/- 11.67 | 8.42 +/- 10.88 | +1.92 | 43.0 | 0.0192 | 0.0385 | **Yes** |
| Ablation (m=4) | +Pseudo (Raw) vs. +Motion (100%) | `Mean_HD95` | 9.19 +/- 12.27 | 10.34 +/- 11.67 | -1.16 | 65.0 | 0.1429 | 0.1429 | No |
| Ablation (m=4) | +Conf Filter (Full) vs. +Pseudo Raw (100%) | `Mean_HD95` | 13.26 +/- 12.78 | 9.19 +/- 12.27 | +4.08 | 17.0 | 0.0004 | 0.0012 | **Yes** |
| Ablation (m=4) | Alt. (SSL+Filtered) vs. Full Pipeline (100%) | `Mean_HD95` | 7.17 +/- 8.33 | 13.26 +/- 12.78 | -6.09 | 8.0 | 4.77e-05 | 0.0002 | **Yes** |
| Robustness vs. Clean (m=4) | Full Noise (std=0.05) vs. Clean | `Mean_HD95` | 16.60 +/- 21.07 | 13.26 +/- 12.78 | +3.34 | 104.0 | 0.9854 | 1.0000 | No |
| Robustness vs. Clean (m=4) | Full Noise (std=0.10) vs. Clean | `Mean_HD95` | 13.73 +/- 14.86 | 13.26 +/- 12.78 | +0.47 | 84.0 | 0.4524 | 1.0000 | No |
| Robustness vs. Clean (m=4) | Full Scale (0.9x) vs. Clean | `Mean_HD95` | 12.47 +/- 12.71 | 13.26 +/- 12.78 | -0.80 | 92.0 | 0.6477 | 1.0000 | No |
| Robustness vs. Clean (m=4) | Full Scale (1.1x) vs. Clean | `Mean_HD95` | 13.94 +/- 13.62 | 13.26 +/- 12.78 | +0.68 | 64.0 | 0.1327 | 0.5309 | No |
| Robustness Full vs. Sup. (m=5) | Clean: Full vs. Sup. (handoff) | `Mean_HD95` | 13.26 +/- 12.78 | 21.98 +/- 16.96 | -8.72 | 13.0 | 0.0002 | 0.0005 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Noise (0.05): Full vs. Sup. (handoff) | `Mean_HD95` | 16.60 +/- 21.07 | 20.32 +/- 18.42 | -3.72 | 38.0 | 0.0107 | 0.0107 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Noise (0.10): Full vs. Sup. (handoff) | `Mean_HD95` | 13.73 +/- 14.86 | 21.91 +/- 19.66 | -8.18 | 12.0 | 0.0001 | 0.0005 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Scale (0.9x): Full vs. Sup. (handoff) | `Mean_HD95` | 12.47 +/- 12.71 | 20.75 +/- 15.90 | -8.28 | 12.0 | 0.0001 | 0.0005 | **Yes** |
| Robustness Full vs. Sup. (m=5) | Scale (1.1x): Full vs. Sup. (handoff) | `Mean_HD95` | 13.94 +/- 13.62 | 22.87 +/- 18.01 | -8.93 | 11.0 | 0.0001 | 0.0005 | **Yes** |
