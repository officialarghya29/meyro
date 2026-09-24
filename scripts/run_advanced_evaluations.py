"""Run Full Advanced Model Evaluation Suite: Ablation, Robustness, Cold Start."""

import json

from meyro.evaluation.ablation import AblationStudy
from meyro.evaluation.cold_start import ColdStartAnalysis
from meyro.evaluation.robustness import RobustnessTestSuite


def main():
    print("=" * 70)
    print("MEYRO ADVANCED MODEL EVALUATION SUITE")
    print("=" * 70)

    # 1. Ablation Study
    print("\n[1/3] EXECUTING ABLATION STUDY (Phase 19)...")
    ablation = AblationStudy(n_subjects=12, n_days=50, seed=42)
    ablation_res = ablation.run()
    print(json.dumps(ablation_res, indent=2))

    # 2. Robustness Stress Tests
    print("\n[2/3] EXECUTING ROBUSTNESS STRESS SUITE (Phase 20)...")
    rob = RobustnessTestSuite(seed=42)
    missing_res = rob.run_missing_data_stress()
    noise_res = rob.run_noise_stress()
    print("Missing Data Stress AUROC:")
    print(json.dumps(missing_res, indent=2))
    print("Sensor Noise Stress AUROC:")
    print(json.dumps(noise_res, indent=2))

    # 3. Few-shot Personalization & Cold Start
    print("\n[3/3] EXECUTING FEW-SHOT COLD START CURVE (Phases 21-23)...")
    cold = ColdStartAnalysis(seed=42)
    cold_res = cold.run_curve()
    print(json.dumps(cold_res, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
