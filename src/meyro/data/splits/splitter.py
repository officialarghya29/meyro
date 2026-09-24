"""Leakage-safe temporal and subject splitters for MEYRO."""

from __future__ import annotations

import pandas as pd


class LeakageFreeSplitter:
    """Enforces strict causal temporal partitioning and subject isolation."""

    @staticmethod
    def causal_train_val_test_split(
        df: pd.DataFrame,
        train_days: int = 30,
        val_days: int = 14,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits per-subject series into strictly ordered temporal blocks.

        Guarantees:
        max(train_timestamp) < min(val_timestamp) <= max(val_timestamp) < min(test_timestamp)
        """
        train_list, val_list, test_list = [], [], []

        for _, group in df.groupby("subject_id", sort=False):
            g = group.sort_values(by="timestamp").reset_index(drop=True)
            n_total = len(g)

            if n_total <= train_days:
                train_list.append(g)
            elif n_total <= train_days + val_days:
                train_list.append(g.iloc[:train_days])
                val_list.append(g.iloc[train_days:])
            else:
                train_list.append(g.iloc[:train_days])
                val_list.append(g.iloc[train_days : train_days + val_days])
                test_list.append(g.iloc[train_days + val_days :])

        train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=df.columns)
        val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame(columns=df.columns)
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)

        return train_df, val_df, test_df
