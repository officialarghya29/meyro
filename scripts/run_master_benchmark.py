"""Run the master model comparison (Phase 18) and persist results."""

from __future__ import annotations

import json
import sys

from meyro.experiments.benchmark import MasterModelBenchmark
from meyro.utils.io import save_experiment_results, set_global_seed

SEED = 42


def main() -> int:
    set_global_seed(SEED)
    benchmark = MasterModelBenchmark(n_subjects=15, n_days=60, calibration_days=21, seed=SEED, epochs=30)
    results = benchmark.run()

    metadata = results.pop("_metadata")

    print("=" * 96)
    print("MEYRO MASTER MODEL COMPARISON — all neural models trained on identical calibration data")
    print("=" * 96)
    header = f"{'Model Architecture':<36} | {'AUROC':>6} | {'AUPRC':>6} | {'F1':>6} | {'FPR@85%Sens':>11}"
    print(header)
    print("-" * 96)
    for name, metrics in results.items():
        print(
            f"{name:<36} | {metrics['auroc']:>6} | {metrics['auprc']:>6} | "
            f"{metrics['f1']:>6} | {metrics['fpr_at_target_sensitivity']:>11}"
        )
    print("=" * 96)
    print(json.dumps(metadata, indent=2))

    save_experiment_results("master_benchmark", {"models": results, "metadata": metadata}, seed=SEED)
    print("\nSaved -> experiments/master_benchmark/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
