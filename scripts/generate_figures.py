"""Generate ultra-futuristic dark-mode scientific figures for MEYRO README & Paper."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Futuristic Palette matching MEYRO Logo: Navy #000C24, Cyan #48D8F0, Mint #90F0C0, Coral #FF5370
plt.style.use("dark_background")


def generate_benchmark_figure():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_facecolor("#000C24")
    ax.set_facecolor("#05112B")

    models = [
        "Population Baseline",
        "Isolation Forest",
        "One-Class SVM",
        "Personal Baseline (Adapt)",
        "MEYRO Personal (Ours)",
    ]
    fpr_values = [19.07, 14.40, 1.56, 1.36, 0.78]
    colors = ["#FF5370", "#FF9100", "#48D8F0", "#64FFDA", "#90F0C0"]

    bars = ax.barh(models, fpr_values, color=colors, height=0.55, edgecolor="#ffffff", linewidth=0.5)

    ax.set_xlabel("False Alarm Rate (FPR % at 85% Sensitivity) — Lower is Better", fontsize=11, color="#48D8F0", labelpad=10)
    ax.set_title("MEYRO MASTER BENCHMARK: 24.4x FALSE ALARM REDUCTION", fontsize=13, fontweight="bold", color="#90F0C0", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.2, color="#48D8F0")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#ffffff",
        )

    ax.set_xlim(0, 23)
    plt.tight_layout()
    plt.savefig("assets/benchmark_fpr_comparison.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def generate_concept_timeline():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=300)
    fig.patch.set_facecolor("#000C24")
    ax1.set_facecolor("#05112B")
    ax2.set_facecolor("#05112B")

    days = np.arange(1, 41)
    # Healthy baseline: 58 bpm
    hr = 58.0 + np.random.normal(0, 1.8, len(days))
    # Illness event between day 20 and 26: HR elevates to 76 bpm
    hr[19:26] += 18.0

    # Plot 1: Standard Population Perspective
    ax1.plot(days, hr, color="#48D8F0", lw=2, label="User Observed Resting HR")
    ax1.axhspan(60, 100, color="#64FFDA", alpha=0.1, label="Population 'Normal' Band (60-100 bpm)")
    ax1.set_title("TRADITIONAL POPULATION THRESHOLD (Fails to detect 18 bpm illness elevation)", fontsize=11, color="#FF5370", fontweight="bold")
    ax1.set_ylabel("Resting HR (bpm)", fontsize=10, color="#48D8F0")
    ax1.set_ylim(45, 95)
    ax1.grid(alpha=0.15, color="#48D8F0")
    ax1.legend(loc="upper left", facecolor="#000C24", edgecolor="#48D8F0", fontsize=9)

    # Plot 2: MEYRO Personal Normal Perspective
    ax2.plot(days, hr, color="#48D8F0", lw=2, label="User Observed Resting HR")
    ax2.axhspan(54, 62, color="#90F0C0", alpha=0.2, label="MEYRO Personal Normal Band (54-62 bpm)")
    ax2.axvspan(20, 26, color="#FF5370", alpha=0.25, label="DETECTED DEVIATION (Personal Outlier)")
    ax2.set_title("MEYRO PERSONALIZED BASELINE (Immediately flags personal deviation)", fontsize=11, color="#90F0C0", fontweight="bold")
    ax2.set_xlabel("Longitudinal Timeline (Days)", fontsize=10, color="#48D8F0")
    ax2.set_ylabel("Resting HR (bpm)", fontsize=10, color="#48D8F0")
    ax2.set_ylim(45, 95)
    ax2.grid(alpha=0.15, color="#48D8F0")
    ax2.legend(loc="upper left", facecolor="#000C24", edgecolor="#90F0C0", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/meyro_concept_timeline.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


if __name__ == "__main__":
    generate_benchmark_figure()
    generate_concept_timeline()
    print("Scientific figures generated successfully in assets/")
