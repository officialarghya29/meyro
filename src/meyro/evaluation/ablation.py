"""Ablation study suite for MEYRO (Phase 19).

Systematically evaluates:
- MEYRO Full Model
- WITHOUT Personal Memory (B_t = 0)
- WITHOUT Context Encoder (C_t = 0)
- WITHOUT Quality Conditioning (Q_t = 1)
- WITHOUT Relational Deviation Module (Naive Euclidean ||E_t - B_t||)
- WITHOUT Adaptive Memory Update (Fixed static memory)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from meyro.data.synthetic import SyntheticBenchmarkGenerator
from meyro.models.meyro import MEYROModel
from meyro.preprocessing.pipeline import PreprocessingPipeline


class AblationStudy:
    """Executes controlled ablations on the MEYRO architecture."""

    def __init__(self, n_subjects: int = 12, n_days: int = 50, seed: int = 42) -> None:
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.seed = seed
        self.features = ["step_count", "resting_hr", "sleep_duration_min"]
        self.window_size = 7

    def run(self) -> dict[str, dict[str, float]]:
        gen = SyntheticBenchmarkGenerator(seed=self.seed)
        df = gen.generate_cohort(n_subjects=self.n_subjects, n_days=self.n_days, anomaly_rate_per_subject=0.6)

        pipe = PreprocessingPipeline(feature_cols=self.features)
        cleaned = pipe.clean_and_impute(df)
        calib_df, eval_df = PreprocessingPipeline.create_subject_temporal_splits(cleaned, calibration_days=18)

        # Build sliding windows for evaluation
        eval_windows, eval_labels, contexts, qualities = [], [], [], []
        for _, grp in eval_df.groupby("subject_id", sort=False):
            g = grp.sort_values(by="timestamp").reset_index(drop=True)
            feats = g[self.features].to_numpy()
            lbls = g["ground_truth_label"].to_numpy()
            quals = np.stack([g["quality_score"].to_numpy()] * len(self.features), axis=-1)
            dows = np.array([ts.weekday() for ts in g["timestamp"]])
            ctx = np.stack([np.sin(2 * np.pi * dows / 7.0), np.cos(2 * np.pi * dows / 7.0)], axis=-1)

            wins = PreprocessingPipeline.create_sliding_windows(feats, window_size=self.window_size)
            if len(wins) > 0:
                eval_windows.append(wins)
                eval_labels.append(lbls[self.window_size - 1 :])
                contexts.append(ctx[self.window_size - 1 :])
                qualities.append(quals[self.window_size - 1 :])

        x_eval = torch.tensor(np.concatenate(eval_windows, axis=0), dtype=torch.float32)
        y_true = np.concatenate(eval_labels, axis=0)
        c_eval = torch.tensor(np.concatenate(contexts, axis=0), dtype=torch.float32)
        q_eval = torch.tensor(np.concatenate(qualities, axis=0), dtype=torch.float32)

        # Compute per-subject personal baseline memory from calibration
        base_mem_list = []
        model = MEYROModel(input_dim=len(self.features), context_dim=2, quality_dim=len(self.features), hidden_dim=16)
        model.eval()

        # Subject-specific baseline reference
        subj_bases = {}
        with torch.no_grad():
            for subj_id, grp in calib_df.groupby("subject_id", sort=False):
                g = grp.sort_values(by="timestamp").reset_index(drop=True)
                feats = g[self.features].to_numpy()
                dows = np.array([ts.weekday() for ts in g["timestamp"]])
                ctx = np.stack([np.sin(2 * np.pi * dows / 7.0), np.cos(2 * np.pi * dows / 7.0)], axis=-1)
                w = PreprocessingPipeline.create_sliding_windows(feats, window_size=self.window_size)
                if len(w) > 0:
                    tw = torch.tensor(w, dtype=torch.float32)
                    tc = torch.tensor(ctx[self.window_size - 1 :], dtype=torch.float32)
                    emb = model.encode_observation(tw, tc)
                    subj_bases[subj_id] = torch.mean(emb, dim=0, keepdim=True)

        for _, grp in eval_df.groupby("subject_id", sort=False):
            subj_id = grp["subject_id"].iloc[0]
            b_val = subj_bases.get(subj_id, torch.zeros(1, 16))
            n_wins = max(0, len(grp) - self.window_size + 1)
            base_mem_list.append(b_val.repeat(n_wins, 1))

        b_eval = torch.cat(base_mem_list, dim=0)

        results = {}

        # 1. Full MEYRO Model
        with torch.no_grad():
            a_full, _, _, _ = model(x_eval, c_eval, b_eval, q_eval)
            results["MEYRO Full Model"] = self._metrics(y_true, a_full.squeeze(-1).numpy())

        # 2. Ablation: Without Personal Memory (B_t = 0)
        with torch.no_grad():
            b_zero = torch.zeros_like(b_eval)
            a_no_mem, _, _, _ = model(x_eval, c_eval, b_zero, q_eval)
            results["Ablation: w/o Personal Memory (B_t=0)"] = self._metrics(y_true, a_no_mem.squeeze(-1).numpy())

        # 3. Ablation: Without Context (C_t = 0)
        with torch.no_grad():
            c_zero = torch.zeros_like(c_eval)
            a_no_ctx, _, _, _ = model(x_eval, c_zero, b_eval, q_eval)
            results["Ablation: w/o Context (C_t=0)"] = self._metrics(y_true, a_no_ctx.squeeze(-1).numpy())

        # 4. Ablation: Without Quality Conditioning (Q_t = 1)
        with torch.no_grad():
            q_ones = torch.ones_like(q_eval)
            a_no_q, _, _, _ = model(x_eval, c_eval, b_eval, q_ones)
            results["Ablation: w/o Quality Conditioning"] = self._metrics(y_true, a_no_q.squeeze(-1).numpy())

        # 5. Ablation: Naive Euclidean Distance ||E_t - B_t|| (No Relational Deviation Net)
        with torch.no_grad():
            e_t = model.encode_observation(x_eval, c_eval)
            dist = torch.norm(e_t - b_eval, p=2, dim=-1)
            results["Ablation: Naive Euclidean (No Deviation Net)"] = self._metrics(y_true, dist.numpy())

        return results

    @staticmethod
    def _metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
        if len(np.unique(y_true)) < 2:
            return {"auroc": 0.5, "auprc": 0.0, "fpr_at_85": 1.0}
        auroc = float(roc_auc_score(y_true, y_scores))
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        s_idx = np.argsort(rec)
        auprc = float(np.trapezoid(prec[s_idx], rec[s_idx]) if hasattr(np, "trapezoid") else np.trapz(prec[s_idx], rec[s_idx]))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        t_idx = np.where(tpr >= 0.85)[0]
        fpr_85 = float(fpr[t_idx[0]]) if len(t_idx) > 0 else 1.0
        return {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "fpr_at_85": round(fpr_85, 4),
        }
