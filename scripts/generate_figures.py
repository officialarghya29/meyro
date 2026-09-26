"""Generate README/paper figures from the committed experiment artifacts.

Every plotted value is read from ``experiments/<suite>/results.json``, which is
produced by ``scripts/run_full_evaluation.py``. No number is typed by hand, so a
figure can never drift away from the result it claims to show.

Run ``scripts/run_full_evaluation.py`` first. If an artifact is missing, the
script fails loudly rather than inventing a plausible-looking chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

# Palette sampled from the MEYRO logo.
NAVY = "#000C24"
PANEL = "#05112B"
CYAN = "#48D8F0"
MINT = "#90F0C0"
CORAL = "#FF5370"
AMBER = "#FFB74D"

EXPERIMENTS = Path("experiments")
ASSETS = Path("assets")


def load_artifact(suite: str) -> dict:
    """Loads a suite artifact, failing loudly when it has not been generated."""
    path = EXPERIMENTS / suite / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment artifact: {path}. Run scripts/run_full_evaluation.py first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["results"]


def _frame(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=13, fontweight="bold", color=MINT, pad=18)
    if subtitle:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color=CYAN,
        )
    return fig, ax


def _horizontal_bars(ax, labels, values, colors, xlabel, value_fmt="{:.3f}", xlim=None):
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor="#ffffff", linewidth=0.4)
    ax.set_xlabel(xlabel, fontsize=11, color=CYAN, labelpad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.2, color=CYAN)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + (max(values) * 0.015 if max(values) else 0.01),
            bar.get_y() + bar.get_height() / 2,
            value_fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#ffffff",
        )
    if xlim:
        ax.set_xlim(*xlim)


def benchmark_figures() -> None:
    results = load_artifact("master_benchmark")
    models = results["models"]

    order = sorted(models, key=lambda name: models[name]["auroc"])
    labels = [name.replace(" (trained)", "").replace(" (ours,", " (") for name in order]

    # AUROC
    fig, ax = _frame(
        "MASTER MODEL COMPARISON — AUROC",
        "All neural models trained on identical calibration windows; streamed causally",
    )
    values = [models[name]["auroc"] for name in order]
    colors = [
        MINT if "MEYRO" in name else CORAL if "untrained" in name or "Population" in name else CYAN
        for name in order
    ]
    _horizontal_bars(ax, labels, values, colors, "AUROC (higher is better)", "{:.3f}", (0, 1.05))
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_auroc.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # FPR at matched sensitivity
    fig, ax = _frame(
        "FALSE-POSITIVE RATE AT 85% SENSITIVITY",
        "Models compared at equal recall — lower is better",
    )
    values = [models[name]["fpr_at_target_sensitivity"] for name in order]
    colors = [
        MINT if "MEYRO" in name else CORAL if "untrained" in name or "Population" in name else CYAN
        for name in order
    ]
    _horizontal_bars(
        ax,
        labels,
        values,
        colors,
        "False-positive rate @ 85% sensitivity (lower is better)",
        "{:.3f}",
        (0, max(values) * 1.2),
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_fpr_comparison.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def ablation_figures() -> None:
    results = load_artifact("ablation")["ablations"]
    order = sorted(results, key=lambda name: results[name]["auroc"])
    labels = [name.replace("Ablation: ", "").replace("Reference: ", "REF ") for name in order]
    values = [results[name]["auroc"] for name in order]
    colors = [MINT if "Full" in name else CORAL if "Untrained" in name else CYAN for name in order]

    fig, ax = _frame(
        "ABLATION STUDY — MEYRO-V2",
        "Every variant streamed causally on identical evaluation windows",
    )
    _horizontal_bars(ax, labels, values, colors, "AUROC (higher is better)", "{:.3f}", (0, 1.05))
    fig.tight_layout()
    fig.savefig(ASSETS / "ablation_auroc.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def cold_start_figure() -> None:
    results = load_artifact("cold_start")
    days = sorted(int(key.split("_")[0]) for key in results)
    pop = [results[f"{d}_days"]["population_auroc"] for d in days]
    pers = [results[f"{d}_days"]["personalized_auroc"] for d in days]

    fig, ax = _frame(
        "FEW-SHOT PERSONALIZATION CURVE",
        "How much personal history before a personal baseline pays off",
    )
    ax.plot(days, pers, "-o", color=MINT, lw=2.5, label="Personalized baseline")
    ax.plot(days, pop, "-s", color=CORAL, lw=2.5, label="Population baseline")
    ax.set_xlabel("Calibration history (days)", fontsize=11, color=CYAN, labelpad=10)
    ax.set_ylabel("AUROC", fontsize=11, color=CYAN)
    ax.grid(alpha=0.2, color=CYAN)
    ax.legend(facecolor=NAVY, edgecolor=CYAN, fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "cold_start_curve.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def drift_figure() -> None:
    results = load_artifact("drift")
    methods = [
        key
        for key in results
        if key
        in {
            "Personal Baseline (Static)",
            "Personal Baseline (Adaptive)",
            "MEYRO-V2 (streamed, trained)",
        }
    ]
    values = [results[m]["persistent_false_alarm_rate"] * 100 for m in methods]
    colors = [MINT if "MEYRO" in m else CORAL if "Static" in m else CYAN for m in methods]

    fig, ax = _frame(
        "BASELINE DRIFT — PERSISTENT FALSE ALARMS",
        "Share of post-drift steady-state days still alerted after a sustained lifestyle change",
    )
    bars = ax.bar(methods, values, color=colors, width=0.55, edgecolor="#ffffff", linewidth=0.4)
    ax.set_ylabel("Persistent false-alarm rate (%) — lower is better", fontsize=11, color=CYAN)
    ax.grid(axis="y", linestyle="--", alpha=0.2, color=CYAN)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#ffffff",
        )
    ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
    fig.autofmt_xdate(rotation=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "baseline_drift.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def robustness_figure() -> None:
    results = load_artifact("robustness")
    axes_specs = [
        ("missing_data", "Missing data (%)", "MISSING DATA"),
        ("sensor_noise", "Noise factor", "SENSOR NOISE"),
        ("spike_outliers", "Outlier rate (%)", "SPIKE OUTLIERS"),
        ("history_length", "History (days)", "HISTORY LENGTH"),
    ]

    fig, grid = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    fig.patch.set_facecolor(NAVY)
    for ax, (axis_key, xlabel, title) in zip(grid.ravel(), axes_specs, strict=True):
        entries = results[axis_key]
        x = list(range(len(entries)))
        y = list(entries.values())
        ax.set_facecolor(PANEL)
        ax.plot(x, y, "-o", color=CYAN, lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(list(entries.keys()), fontsize=8, rotation=15)
        ax.set_title(title, fontsize=11, color=MINT, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9, color=CYAN)
        ax.set_ylabel("AUROC", fontsize=9, color=CYAN)
        ax.grid(alpha=0.2, color=CYAN)
        ax.set_ylim(0.9, 1.0)
    fig.suptitle(
        "ROBUSTNESS OF THE PERSONAL STATISTICAL BASELINE",
        fontsize=13,
        color=MINT,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "robustness_stress.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def significance_figure() -> None:
    results = load_artifact("significance")
    methods = list(results["per_method"].keys())
    means = [results["per_method"][m]["per_subject_auroc"]["mean"] for m in methods]
    lowers = [results["per_method"][m]["per_subject_auroc"]["ci_lower"] for m in methods]
    uppers = [results["per_method"][m]["per_subject_auroc"]["ci_upper"] for m in methods]
    colors = [CORAL if "Population" in m else MINT for m in methods]

    fig, ax = _frame(
        "PER-SUBJECT AUROC WITH 95% BOOTSTRAP CIs",
        "Unit of analysis is the subject; test is a paired Wilcoxon signed-rank across seeds",
    )
    y = np.arange(len(methods))
    errors = [
        [mean - lower for mean, lower in zip(means, lowers, strict=True)],
        [upper - mean for mean, upper in zip(means, uppers, strict=True)],
    ]
    ax.errorbar(means, y, xerr=errors, fmt="o", color=CYAN, ecolor=MINT, capsize=6, markersize=9)
    for index, mean in enumerate(means):
        ax.text(mean, index - 0.22, f"{mean:.4f}", ha="center", fontsize=9, color="#ffffff")
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Per-subject AUROC (95% CI)", fontsize=11, color=CYAN)
    ax.grid(axis="x", linestyle="--", alpha=0.2, color=CYAN)
    for index, color in enumerate(colors):
        ax.get_yticklabels()[index].set_color(color)

    test = results["tests"].get("personalized_static_vs_population", {})
    if test:
        ax.set_title(
            "PER-SUBJECT AUROC WITH 95% BOOTSTRAP CIs\n"
            f"personalized vs population: p = {test['p_value']:.2e}, win rate = {test['win_rate_a']:.2f}",
            fontsize=12,
            fontweight="bold",
            color=MINT,
            pad=18,
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "significance_forest.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def neural_gap_figure() -> None:
    results = load_artifact("neural_gap")
    diagnosis = results["diagnosis"]
    budgets = sorted(int(k) for k in diagnosis["epoch_budget_aurocs"])
    values = [diagnosis["epoch_budget_aurocs"][str(b)] for b in budgets]
    control = results["configurations"]["Control: Statistical Personal Baseline"]["auroc"]

    fig, ax = _frame(
        "NEURAL GAP DIAGNOSIS",
        "Training budget is a real constraint; memory adaptation is not the cause of the gap",
    )
    ax.plot(budgets, values, "-o", color=MINT, lw=2.5, markersize=8, label="MEYRO-V2 (streaming)")
    ax.axhline(
        control, color=CORAL, ls="--", lw=2, label=f"Statistical personal baseline ({control:.4f})"
    )
    frozen = results["configurations"]["MEYRO-V2 frozen memory"]["auroc"]
    streaming = results["configurations"]["MEYRO-V2 streaming (adaptive)"]["auroc"]
    ax.scatter([budgets[0]], [frozen], color=CYAN, zorder=5, s=90, marker="s")
    ax.annotate(
        f"frozen memory {frozen:.4f}\n(adaptation cost {frozen - streaming:+.4f})",
        xy=(budgets[0], frozen),
        xytext=(budgets[0] + 8, frozen - 0.03),
        color=CYAN,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": CYAN},
    )
    for budget, value in zip(budgets, values, strict=True):
        ax.text(budget, value + 0.004, f"{value:.4f}", ha="center", fontsize=9, color="#ffffff")
    ax.set_xlabel("Training epochs", fontsize=11, color=CYAN, labelpad=10)
    ax.set_ylabel("AUROC", fontsize=11, color=CYAN)
    ax.grid(alpha=0.2, color=CYAN)
    ax.legend(facecolor=NAVY, edgecolor=CYAN, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "neural_gap_diagnosis.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def cross_subject_figure() -> None:
    results = load_artifact("cross_subject")
    labels = [
        "MEYRO-V2\nwithin-subject",
        "MEYRO-V2\nunseen subjects",
        "Statistical baseline\nunseen subjects",
    ]
    values = [
        results["within_subject"]["MEYRO-V2 streaming"]["auroc"],
        results["cross_subject"]["MEYRO-V2 streaming (unseen subjects)"]["auroc"],
        results["cross_subject"]["Statistical Personal Baseline (unseen subjects)"]["auroc"],
    ]
    colors = [CYAN, MINT, CORAL]

    fig, ax = _frame(
        "HELD-OUT SUBJECT GENERALIZATION",
        "Held-out subjects never contribute to training or to the standardizer",
    )
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#ffffff", linewidth=0.4)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#ffffff",
        )
    ax.set_ylabel("AUROC", fontsize=11, color=CYAN)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.2, color=CYAN)
    ax.text(
        0.5,
        0.06,
        f"generalization gap: {results['generalization_gap_auroc']:+.4f} AUROC",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color=AMBER,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "cross_subject.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def concept_timeline() -> None:
    """Illustrative schematic of the core idea.

    This is a conceptual illustration with a fixed seed, NOT measured data. It is
    labelled as such on the figure so it cannot be mistaken for a result.
    """
    rng = np.random.default_rng(0)
    days = np.arange(1, 41)
    resting_hr = 58.0 + rng.normal(0, 1.8, len(days))
    resting_hr[19:26] += 18.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=300)
    fig.patch.set_facecolor(NAVY)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.plot(days, resting_hr, color=CYAN, lw=2, label="Illustrative observation")
        ax.set_ylabel("Resting HR (bpm)", fontsize=10, color=CYAN)
        ax.set_ylim(45, 95)
        ax.grid(alpha=0.15, color=CYAN)

    ax1.axhspan(60, 100, color=MINT, alpha=0.10, label="Population 'normal' band")
    ax1.set_title(
        "POPULATION THRESHOLD — an 18 bpm personal excursion stays inside the population band",
        fontsize=11,
        color=CORAL,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", facecolor=NAVY, edgecolor=CYAN, fontsize=9)

    ax2.axhspan(54, 62, color=MINT, alpha=0.2, label="Personal baseline band")
    ax2.axvspan(20, 26, color=CORAL, alpha=0.25, label="Deviation from personal baseline")
    ax2.set_title(
        "PERSONAL BASELINE — the same excursion exceeds this person's own normal range",
        fontsize=11,
        color=MINT,
        fontweight="bold",
    )
    ax2.set_xlabel("Illustrative timeline (days)", fontsize=10, color=CYAN)
    ax2.legend(loc="upper left", facecolor=NAVY, edgecolor=MINT, fontsize=9)

    fig.suptitle(
        "CONCEPTUAL ILLUSTRATION (synthetic, fixed seed) — not a measured result",
        fontsize=10,
        color=AMBER,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "meyro_concept_timeline.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    benchmark_figures()
    ablation_figures()
    cold_start_figure()
    drift_figure()
    robustness_figure()
    significance_figure()
    neural_gap_figure()
    cross_subject_figure()
    concept_timeline()
    print("Figures written to assets/ (all values read from experiments/*/results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
