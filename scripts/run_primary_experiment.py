"""Run Primary MEYRO Research Experiment: Population Baseline vs. Personalized Baseline."""

import json

from meyro.experiments.primary import PrimaryExperiment


def main():
    print("=" * 65)
    print("MEYRO PRIMARY RESEARCH EXPERIMENT (Phase 11)")
    print("Research Question: Can personalized baselines detect meaningful")
    print("longitudinal deviations more effectively than population baselines?")
    print("=" * 65)

    exp = PrimaryExperiment(n_subjects=20, n_days=90, calibration_days=28, seed=42)
    results = exp.run()

    print("\n[RESULTS SUMMARY]")
    print(json.dumps(results, indent=2))

    pop = results["population"]
    pers = results["personalized_static"]
    adapt = results["personalized_adaptive"]

    print("\n[COMPARATIVE METRICS]")
    print(f"Cohort Size: {exp.n_subjects} subjects, {exp.n_days} days longitudinal horizon")
    print(f"Evaluation Windows: {results['n_evaluation_samples']} total ({results['n_positive_deviations']} positive anomaly states)\n")
    print(f"{'Condition':<25} | {'AUROC':<8} | {'AUPRC':<8} | {'FPR @ 85% Sens':<15}")
    print("-" * 65)
    print(f"{'Population Baseline':<25} | {pop['auroc']:<8} | {pop['auprc']:<8} | {pop['fpr_at_85_recall']:<15}")
    print(f"{'Personalized (Static)':<25} | {pers['auroc']:<8} | {pers['auprc']:<8} | {pers['fpr_at_85_recall']:<15}")
    print(f"{'Personalized (Adaptive)':<25} | {adapt['auroc']:<8} | {adapt['auprc']:<8} | {adapt['fpr_at_85_recall']:<15}")
    print("=" * 65)


if __name__ == "__main__":
    main()
