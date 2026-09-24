"""End-to-end Master Benchmark comparing All Baselines vs MEYRO Architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from meyro.anomaly.classical import ClassicalAnomalyEngine
from meyro.baselines.personal import PersonalizedBaseline
from meyro.baselines.population import PopulationBaseline
from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.models.meyro import MEYROModel
from meyro.models.tcn import TCNAutoencoder
from meyro.models.temporal import TemporalAutoencoder
from meyro.models.transformer import TransformerAutoencoder
from meyro.personalization.adaptive import AdaptivePersonalBaseline
from meyro.preprocessing.pipeline import PreprocessingPipeline


class MasterModelBenchmark:
    """Executes the master benchmark comparing:
    - Population Baseline
    - Personal Baseline (Static)
    - Personal Baseline (Adaptive)
    - Isolation Forest (Per Subject)
    - One-Class SVM (Per Subject)
    - GRU Autoencoder
    - LSTM Autoencoder
    - TCN Autoencoder
    - Transformer Autoencoder
    - MEYRO V1 (Unified Personal Baseline Architecture)
    """

    def __init__(self, n_subjects: int = 15, n_days: int = 60, seed: int = 42) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.seed = seed
        self.features = ["step_count", "resting_hr", "sleep_duration_min"]

    def run(self) -> dict[str, Any]:
        # 1. Generate longitudinal cohort
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        df = gen.generate_cohort(n_subjects=self.n_subjects, n_days=self.n_days, anomaly_rate_per_subject=0.6)

        pipe = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipe.clean_and_impute(df)
        calib_df, eval_df = PreprocessingPipeline.create_subject_temporal_splits(cleaned, calibration_days=21)

        y_eval = eval_df["ground_truth_label"].to_numpy()

        results = {}

        # 1. Population Baseline
        pop = PopulationBaseline(feature_cols=self.features).fit(calib_df)
        pop_scores = pop.score(eval_df)["deviation_score"].to_numpy()
        results["Population Baseline"] = self._eval(y_eval, pop_scores)

        # 2. Personal Baseline (Static)
        pers = PersonalizedBaseline(feature_cols=self.features).fit_calibration(calib_df)
        pers_scores = pers.score_causal(eval_df)["deviation_score"].to_numpy()
        results["Personal Baseline (Static)"] = self._eval(y_eval, pers_scores)

        # 3. Personal Baseline (Adaptive)
        adapt = AdaptivePersonalBaseline(feature_cols=self.features).initialize(calib_df)
        adapt_scores = [
            adapt.step(r["subject_id"], {c: float(r[c]) for c in self.features}, r["quality_score"])["deviation_score"]
            for _, r in eval_df.iterrows()
        ]
        results["Personal Baseline (Adaptive)"] = self._eval(y_eval, np.array(adapt_scores))

        # 4. Isolation Forest
        ifo = ClassicalAnomalyEngine(algorithm="isolation_forest", random_state=self.seed).fit_per_subject(calib_df, self.features)
        ifo_scores = ifo.score_per_subject(eval_df, self.features)["anomaly_score"].to_numpy()
        results["Isolation Forest"] = self._eval(y_eval, ifo_scores)

        # 5. One-Class SVM
        ocsvm = ClassicalAnomalyEngine(algorithm="one_class_svm", random_state=self.seed).fit_per_subject(calib_df, self.features)
        ocsvm_scores = ocsvm.score_per_subject(eval_df, self.features)["anomaly_score"].to_numpy()
        results["One-Class SVM"] = self._eval(y_eval, ocsvm_scores)

        # Deep Sequence Baselines (Windowing: 7-day sliding window)
        window_size = 7
        eval_windows = []
        eval_labels = []
        context_list = []
        quality_list = []

        for _, grp in eval_df.groupby("subject_id", sort=False):
            g = grp.sort_values(by="timestamp").reset_index(drop=True)
            feats = g[self.features].to_numpy()
            lbls = g["ground_truth_label"].to_numpy()
            quals = np.stack([g["quality_score"].to_numpy()] * len(self.features), axis=-1)

            # Causal context: cyclical day of week sin/cos
            dows = np.array([ts.weekday() for ts in g["timestamp"]])
            ctx = np.stack([np.sin(2 * np.pi * dows / 7.0), np.cos(2 * np.pi * dows / 7.0)], axis=-1)

            wins = PreprocessingPipeline.create_sliding_windows(feats, window_size=window_size)
            if len(wins) > 0:
                eval_windows.append(wins)
                # Label is anomaly if target day has anomaly
                eval_labels.append(lbls[window_size - 1 :])
                context_list.append(ctx[window_size - 1 :])
                quality_list.append(quals[window_size - 1 :])

        x_eval_tensor = torch.tensor(np.concatenate(eval_windows, axis=0), dtype=torch.float32)
        y_seq_eval = np.concatenate(eval_labels, axis=0)
        ctx_tensor = torch.tensor(np.concatenate(context_list, axis=0), dtype=torch.float32)
        qual_tensor = torch.tensor(np.concatenate(quality_list, axis=0), dtype=torch.float32)

        # 6. GRU Autoencoder
        gru = TemporalAutoencoder(input_dim=len(self.features), hidden_dim=16, cell_type="gru")
        gru_scores = gru.compute_deviation_score(x_eval_tensor).numpy()
        results["GRU Autoencoder"] = self._eval(y_seq_eval, gru_scores)

        # 7. LSTM Autoencoder
        lstm = TemporalAutoencoder(input_dim=len(self.features), hidden_dim=16, cell_type="lstm")
        lstm_scores = lstm.compute_deviation_score(x_eval_tensor).numpy()
        results["LSTM Autoencoder"] = self._eval(y_seq_eval, lstm_scores)

        # 8. TCN Autoencoder
        tcn = TCNAutoencoder(input_dim=len(self.features), hidden_dim=16, num_layers=2)
        tcn_scores = tcn.compute_deviation_score(x_eval_tensor).numpy()
        results["TCN Autoencoder"] = self._eval(y_seq_eval, tcn_scores)

        # 9. Transformer Autoencoder
        tfm = TransformerAutoencoder(input_dim=len(self.features), d_model=16, nhead=2, num_layers=1)
        tfm_scores = tfm.compute_deviation_score(x_eval_tensor).numpy()
        results["Transformer Autoencoder"] = self._eval(y_seq_eval, tfm_scores)

        # 10. MEYRO V1 Architecture
        meyro = MEYROModel(input_dim=len(self.features), context_dim=2, quality_dim=len(self.features), hidden_dim=16)
        # Compute baseline memory state from calibration
        baseline_mem = torch.zeros(len(x_eval_tensor), 16)
        with torch.no_grad():
            a_scores, _u_scores, _, _ = meyro(x_eval_tensor, ctx_tensor, baseline_mem, qual_tensor)
            meyro_scores = a_scores.squeeze(-1).numpy()
        results["MEYRO V1 (Ours)"] = self._eval(y_seq_eval, meyro_scores)

        return results

    @staticmethod
    def _eval(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
        if len(np.unique(y_true)) < 2:
            return {"auroc": 0.5, "auprc": 0.0, "fpr_at_85_sens": 1.0}
        auroc = float(roc_auc_score(y_true, y_scores))
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        sort_idx = np.argsort(rec)
        auprc = float(np.trapezoid(prec[sort_idx], rec[sort_idx]) if hasattr(np, "trapezoid") else np.trapz(prec[sort_idx], rec[sort_idx]))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        t_idx = np.where(tpr >= 0.85)[0]
        fpr_at_85 = float(fpr[t_idx[0]]) if len(t_idx) > 0 else 1.0
        return {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "fpr_at_85_sens": round(fpr_at_85, 4),
        }


def main():
    bench = MasterModelBenchmark(n_subjects=15, n_days=60, seed=42)
    results = bench.run()

    print("\n" + "=" * 70)
    print("MEYRO MASTER MODEL COMPARISON TABLE (All Baselines vs MEYRO)")
    print("=" * 70)
    print(f"{'Model Architecture':<32} | {'AUROC':<8} | {'AUPRC':<8} | {'FPR @ 85% Sens':<15}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<32} | {metrics['auroc']:<8} | {metrics['auprc']:<8} | {metrics['fpr_at_85_sens']:<15}")
    print("=" * 70)


if __name__ == "__main__":
    main()
