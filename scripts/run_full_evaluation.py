"""Run the complete MEYRO evaluation programme and persist every result.

Suites executed (Phases 18-29):

1. Master model comparison    — trained baselines vs MEYRO-V1 / V2
2. Ablation study             — which MEYRO components actually matter
3. Robustness stress tests    — missing data, noise, outliers, history,
                                sampling rate, device variation
4. Cold-start / few-shot      — how much history personalization needs
5. Baseline drift             — sustained lifestyle change vs. transient anomaly
6. Efficiency                 — parameters, latency, model size

Every suite writes ``experiments/<suite>/results.json`` with its configuration,
seed, git commit and environment, so results are traceable and regenerable.
"""

from __future__ import annotations

import json
import sys

import torch

from meyro.evaluation.ablation import AblationStudy
from meyro.evaluation.cold_start import ColdStartAnalysis
from meyro.evaluation.drift import BaselineDriftAnalysis
from meyro.evaluation.efficiency import measure_efficiency
from meyro.evaluation.robustness import RobustnessTestSuite
from meyro.experiments.benchmark import MasterModelBenchmark
from meyro.models.meyro import MEYROModel, MEYROModelV2
from meyro.models.tcn import TCNAutoencoder
from meyro.models.temporal import TemporalAutoencoder
from meyro.models.transformer import TransformerAutoencoder
from meyro.utils.io import save_experiment_results, set_global_seed

SEED = 42


def _print(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_benchmark() -> dict:
    _print("[1/6] MASTER MODEL COMPARISON (Phase 18)")
    results = MasterModelBenchmark(n_subjects=15, n_days=60, seed=SEED, epochs=30).run()
    metadata = results.pop("_metadata")
    for name, metrics in results.items():
        print(
            f"{name:<36} AUROC={metrics['auroc']:<7} AUPRC={metrics['auprc']:<7} "
            f"F1={metrics['f1']:<7} FPR@85={metrics['fpr_at_target_sensitivity']}"
        )
    save_experiment_results("master_benchmark", {"models": results, "metadata": metadata}, seed=SEED)
    return {"models": results, "metadata": metadata}


def run_ablation() -> dict:
    _print("[2/6] ABLATION STUDY (Phase 19) — trained MEYRO-V2")
    results = AblationStudy(n_subjects=12, n_days=50, seed=SEED, epochs=25).run()
    metadata = results.pop("_metadata")
    for name, metrics in results.items():
        print(
            f"{name:<38} AUROC={metrics['auroc']:<7} AUPRC={metrics['auprc']:<7} "
            f"F1={metrics['f1']:<7} FPR@85={metrics['fpr_at_target_sensitivity']}"
        )
    save_experiment_results("ablation", {"ablations": results, "metadata": metadata}, seed=SEED)
    return {"ablations": results, "metadata": metadata}


def run_robustness() -> dict:
    _print("[3/6] ROBUSTNESS STRESS SUITE (Phase 20)")
    suite = RobustnessTestSuite(seed=SEED)
    results = suite.run_all()
    for axis, values in results.items():
        print(f"{axis}: {json.dumps(values)}")
    save_experiment_results("robustness", results, seed=SEED)
    return results


def run_cold_start() -> dict:
    _print("[4/6] FEW-SHOT / COLD-START CURVE (Phases 21-23)")
    results = ColdStartAnalysis(seed=SEED).run_curve()
    print(json.dumps(results, indent=2))
    save_experiment_results("cold_start", results, seed=SEED)
    return results


def run_drift() -> dict:
    _print("[5/6] BASELINE DRIFT (Phase 24)")
    analysis = BaselineDriftAnalysis(seed=SEED)
    results = analysis.run()
    results["acute_outlier_probe"] = analysis.acute_outlier_probe()
    print(json.dumps(results, indent=2))
    save_experiment_results("drift", results, seed=SEED)
    return results


def run_efficiency() -> dict:
    _print("[6/6] MODEL EFFICIENCY (Phase 29)")
    set_global_seed(SEED)
    n_features, window = 3, 7
    x = torch.randn(64, window, n_features)
    ctx = torch.randn(64, 2)
    qual = torch.ones(64, n_features)

    results: dict[str, dict[str, float]] = {}

    gru = TemporalAutoencoder(input_dim=n_features, hidden_dim=16, cell_type="gru")
    results["GRU Autoencoder"] = measure_efficiency(gru, lambda: gru(x))

    lstm = TemporalAutoencoder(input_dim=n_features, hidden_dim=16, cell_type="lstm")
    results["LSTM Autoencoder"] = measure_efficiency(lstm, lambda: lstm(x))

    tcn = TCNAutoencoder(input_dim=n_features, hidden_dim=16, num_layers=2)
    results["TCN Autoencoder"] = measure_efficiency(tcn, lambda: tcn(x))

    transformer = TransformerAutoencoder(input_dim=n_features, d_model=16, nhead=2, num_layers=1)
    results["Transformer Autoencoder"] = measure_efficiency(transformer, lambda: transformer(x))

    v1 = MEYROModel(input_dim=n_features, context_dim=2, quality_dim=n_features, hidden_dim=16)
    zeros = torch.zeros(64, 16)
    results["MEYRO-V1"] = measure_efficiency(v1, lambda: v1(x, ctx, zeros, qual))

    v2 = MEYROModelV2(input_dim=n_features, context_dim=2, quality_dim=n_features, hidden_dim=16)
    results["MEYRO-V2"] = measure_efficiency(v2, lambda: v2(x, ctx, zeros, zeros, qual))

    for name, metrics in results.items():
        print(
            f"{name:<26} params={int(metrics['parameters']):<7} "
            f"latency={metrics['mean_latency_ms']:<8}ms size={metrics['model_size_kb']}KB "
            f"throughput={metrics['throughput_per_sec']}/s"
        )
    save_experiment_results("efficiency", results, seed=SEED)
    return results


def main() -> int:
    set_global_seed(SEED)
    run_benchmark()
    run_ablation()
    run_robustness()
    run_cold_start()
    run_drift()
    run_efficiency()
    _print("ALL SUITES COMPLETE — artifacts written to experiments/*/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
