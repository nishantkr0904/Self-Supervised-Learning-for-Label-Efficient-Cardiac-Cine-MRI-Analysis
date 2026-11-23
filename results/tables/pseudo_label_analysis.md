# Table D: Pseudo-Label Analysis

Analysis of pseudo-label generation quality, confidence filtering rates, and downstream impact on segmentation performance.

| Method / Filtering Configuration | Acceptance Rate (%) | Min Confidence ($p_{max}$) | Max Entropy ($H$) | Downstream Mean Dice | Rejection Rate (%) | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw / Unfiltered Predictions** | 100.0% | 0.00 | None | **0.7898** | 0.0% | Direct teacher argmax without quality filtering |
| **Confidence Filtered (MSP >= 0.90)** | 0.47% | 0.90 | None | **0.7925** | 99.53% | Rejects uncertain background and ambiguous borders |
| **Confidence & Entropy Filtered (MSP >= 0.90, H <= 0.25)** | 0.47% | 0.90 | 0.25 | **0.7941** | 99.53% | Optimal dual-criterion filter (Variant D) |
| **Full Pipeline (+ Motion Temporal Consistency)** | 0.47% | 0.90 | 0.25 | **0.7905** | 99.53% | Joint motion agreement + confidence filtering |

### Filtering Rules & Technical Specifications
- **Confidence Metric**: Maximum Softmax Probability ($p_{max} = \max_c P(y=c|x)$) thresholded at $0.90$.
- **Entropy Metric**: Normalized Shannon Entropy ($H(p) = -\sum p_c \log p_c / \log C$) thresholded at $0.25$.
- **Temporal Consistency Threshold**: Cosine / agreement threshold $0.80$ with motion-warped adjacent frame.
- **Minimum Foreground Area**: Slices with fewer than 30 accepted foreground pixels are rejected (`ignore_index = -1`).

### Data Provenance & Integrity Checksums
- Pseudo-label metadata: `results/pseudo_labels/pseudo_label_metadata.json` (SHA-256: `66f04eec2130b4e9...`)
- Pseudo-label generator module: `src/pseudo_labels.py` (SHA-256: `49d935afdbc0a2e3...`)
- SSL + Pseudo downstream evaluation: `results/experiments/ssl_pseudo_100pct_seed42/test_metrics.json`
