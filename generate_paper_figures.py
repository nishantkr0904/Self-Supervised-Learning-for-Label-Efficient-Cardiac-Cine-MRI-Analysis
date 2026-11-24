#!/usr/bin/env python3
"""
generate_paper_figures.py

Generates publication-quality figures for the research paper:
  - fig2_label_efficiency.pdf: Label-efficiency curves (Mean Dice vs. % Labels)
  - fig3_ablation.pdf: Grouped ablation breakdown across anatomical structures and HD95
  - fig4_baselines.pdf: Comprehensive baseline benchmark comparison
  - fig5_per_patient_distribution.pdf: Per-patient test set Dice distribution (box + scatter)

All data extracted directly from verified experimental outputs.
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "paper_artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

def make_fig2_label_efficiency():
    fractions = [10, 25, 50, 100]
    
    # Verified canonical 20-patient test-set metrics
    ours = [0.6786, 0.7316, 0.7547, 0.7905]
    unet_baseline = [0.5807, 0.6608, 0.7293, 0.8064]
    medsam_finetuned = [0.3382, 0.3838, 0.4398, 0.4897]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    
    ax.plot(fractions, ours, "o-", color="#1f77b4", label="Proposed Full Pipeline (Ours)", lw=2.2, zorder=4)
    ax.plot(fractions, unet_baseline, "s--", color="#d62728", label="Supervised U-Net Baseline", lw=2.0, zorder=3)
    ax.plot(fractions, medsam_finetuned, "^:", color="#ff7f0e", label="MedSAM (Fine-Tuned Decoder)", lw=1.8, zorder=2)
    
    # Annotate key low-label margins
    ax.annotate("+16.9% Rel. Gain\n(0.6786 vs 0.5807)",
                xy=(10, 0.6786), xytext=(11, 0.77),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                fontsize=9.2, fontweight="bold", color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eaf2f8", ec="#1f77b4", lw=0.8))
    
    ax.annotate("+10.7% Rel. Gain\n(0.7316 vs 0.6608)",
                xy=(25, 0.7316), xytext=(28, 0.57),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                fontsize=9.2, fontweight="bold", color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eaf2f8", ec="#1f77b4", lw=0.8))

    ax.annotate("+3.5% Rel. Gain\n(0.7547 vs 0.7293)",
                xy=(50, 0.7547), xytext=(55, 0.64),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                fontsize=9.2, fontweight="bold", color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eaf2f8", ec="#1f77b4", lw=0.8))
    
    ax.set_xlabel("Proportion of Labeled Training Patients (%)")
    ax.set_ylabel("Mean Dice Similarity Coefficient")
    ax.set_title("Label-Efficiency: Segmentation Performance vs. Annotation Budget")
    ax.set_xticks(fractions)
    ax.set_xticklabels(["10%\n(7 pts)", "25%\n(17 pts)", "50%\n(35 pts)", "100%\n(70 pts)"])
    ax.set_ylim(0.25, 0.88)
    ax.legend(loc="lower right", framealpha=0.95)
    
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig2_label_efficiency.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")

def make_fig3_ablation():
    variants = [
        "1. Baseline\n(Supervised)",
        "2. +SSL\n(Temporal)",
        "3. +Motion\n(Warping)",
        "4. +Pseudo\n(Unfiltered)",
        "5. +Conf Filter\n(Full Ours)",
        "Alt. SSL+\nFilt. Pseudo"
    ]
    
    lv = [0.8644, 0.8507, 0.8441, 0.8530, 0.8533, 0.8507]
    myo = [0.7992, 0.7838, 0.7855, 0.7961, 0.7938, 0.7888]
    rv = [0.7554, 0.7348, 0.6871, 0.7113, 0.7244, 0.7428]
    mean_dice = [0.8064, 0.7898, 0.7723, 0.7868, 0.7905, 0.7941]
    hd95 = [6.23, 8.42, 10.34, 9.19, 13.26, 7.17]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3), dpi=300, gridspec_kw={"width_ratios": [2.2, 1]})
    
    x = np.arange(len(variants))
    w = 0.18
    
    ax1.bar(x - 1.5*w, lv, w, label="LV Cavity", color="#3498db")
    ax1.bar(x - 0.5*w, myo, w, label="Myocardium", color="#2ecc71")
    ax1.bar(x + 0.5*w, rv, w, label="RV Cavity", color="#e67e22")
    ax1.bar(x + 1.5*w, mean_dice, w, label="Mean Dice", color="#9b59b6", hatch="//")
    
    ax1.set_ylabel("Dice Similarity Coefficient")
    ax1.set_title("(a) Component Ablation: Per-Class Dice Scores")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=8.8)
    ax1.set_ylim(0.60, 0.95)
    ax1.legend(loc="lower left", ncol=2, framealpha=0.9)
    
    # HD95 subplot
    ax2.bar(x, hd95, color="#e74c3c", width=0.45, edgecolor="black", lw=0.6)
    ax2.set_ylabel("95% Hausdorff Distance (mm)")
    ax2.set_title("(b) Boundary Error (HD95)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=28, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 16)
    
    for i, v in enumerate(hd95):
        ax2.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig3_ablation.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")

def make_fig4_baselines():
    methods = [
        "MedSAM (Zero-Shot)",
        "MedSAM (Fine-Tuned)",
        "Full Pipeline (Ours)",
        "Swin-UNet (Full)",
        "SegFormer (Full)",
        "Supervised U-Net",
        "TransUNet (Full)"
    ]
    
    dice_scores = [0.1790, 0.4897, 0.7905, 0.7994, 0.8038, 0.8064, 0.8189]
    colors = ["#bdc3c7", "#f39c12", "#2980b9", "#16a085", "#27ae60", "#c0392b", "#8e44ad"]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    
    y = np.arange(len(methods))
    bars = ax.barh(y, dice_scores, color=colors, height=0.55, edgecolor="black", lw=0.5)
    
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontweight="medium")
    ax.set_xlabel("Mean Dice Similarity Coefficient (Test Set, 20 Patients)")
    ax.set_title("Benchmarking Architectural Baselines on ACDC Test Cohort")
    ax.set_xlim(0, 1.0)
    
    # Value labels
    for bar, score in zip(bars, dice_scores):
        ax.text(score + 0.015, bar.get_y() + bar.get_height()/2, f"{score:.4f}",
                va="center", fontsize=9.5, fontweight="bold")
        
    ax.axvline(0.80, color="gray", linestyle=":", alpha=0.7)
    
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig4_baselines.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")

def make_fig5_per_patient_distribution():
    # Load per-patient data from full_pipeline_100pct_seed42
    json_path = REPO_ROOT / "results" / "experiments" / "full_pipeline_100pct_seed42" / "test_metrics.json"
    with open(json_path) as f:
        data = json.load(f)
    
    pp = data.get("per_patient", {})
    pids = sorted(pp.keys())
    
    lv_dices = [pp[p]["LV_Dice"] for p in pids]
    myo_dices = [pp[p]["Myocardium_Dice"] for p in pids]
    rv_dices = [pp[p]["RV_Dice"] for p in pids]
    mean_dices = [pp[p]["Mean_Dice"] for p in pids]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    
    box_data = [lv_dices, myo_dices, rv_dices, mean_dices]
    labels = ["LV Cavity", "Myocardium", "RV Cavity", "Patient Mean"]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(color="gray", lw=1.2),
                    capprops=dict(color="gray", lw=1.2),
                    flierprops=dict(marker="o", color="red", alpha=0.5))
    
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
    
    # Overlay individual patient jitter points
    for i, d in enumerate(box_data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.85, s=28, color=colors[i], edgecolor="black", lw=0.5, zorder=5)
    
    ax.set_ylabel("Dice Similarity Coefficient")
    ax.set_title("Per-Patient Dice Distribution: Full Pipeline on 20 Test Patients")
    ax.set_ylim(0.35, 1.02)
    
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig5_per_patient_distribution.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    make_fig2_label_efficiency()
    make_fig3_ablation()
    make_fig4_baselines()
    make_fig5_per_patient_distribution()
    print("All paper figures successfully generated in paper_artifacts/figures/")
