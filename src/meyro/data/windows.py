"""Causal window construction shared by trainers and evaluation suites.

Centralising window creation guarantees that every model (classical, deep, MEYRO)
is evaluated on identical, causality-preserving inputs. Windows are strictly
backward-looking: the window terminating at index ``t`` contains only
observations ``[t - window_size + 1, t]`` and the prediction target is the
observation at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from meyro.preprocessing.pipeline import PreprocessingPipeline

# Cyclical (sin, cos) encoding of the ISO weekday, used as causal context.
CONTEXT_DIM = 2


@dataclass(frozen=True)
class WindowBatch:
    """A materialised batch of causal windows.

    Attributes:
        x: observations, shape ``[N, window_size, n_features]``
        y: binary deviation labels for the window-ending observation, shape ``[N]``
        context: cyclical context per window, shape ``[N, context_dim]``
        quality: per-feature quality scores, shape ``[N, n_features]``
        subject_ids: subject identifier per window, shape ``[N]``
    """

    x: np.ndarray
    y: np.ndarray
    context: np.ndarray
    quality: np.ndarray
    subject_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def context_dim(self) -> int:
        return int(self.context.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[-1])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(x, context, quality)`` as float32 tensors."""
        return (
            torch.tensor(self.x, dtype=torch.float32),
            torch.tensor(self.context, dtype=torch.float32),
            torch.tensor(self.quality, dtype=torch.float32),
        )

    def empty_like(self) -> WindowBatch:
        """Returns a zero-row batch with the same feature/context dimensions."""
        return WindowBatch(
            x=np.empty((0, self.x.shape[1], self.feature_dim), dtype=self.x.dtype),
            y=np.empty((0,), dtype=self.y.dtype),
            context=np.empty((0, self.context_dim), dtype=self.context.dtype),
            quality=np.empty((0, self.feature_dim), dtype=self.quality.dtype),
            subject_ids=np.empty((0,), dtype=self.subject_ids.dtype),
        )

    def concat(self, others: WindowBatch) -> WindowBatch:
        """Concatenates another batch along the sample axis."""
        if len(self) == 0:
            return others
        if len(others) == 0:
            return self
        return WindowBatch(
            x=np.concatenate([self.x, others.x], axis=0),
            y=np.concatenate([self.y, others.y], axis=0),
            context=np.concatenate([self.context, others.context], axis=0),
            quality=np.concatenate([self.quality, others.quality], axis=0),
            subject_ids=np.concatenate([self.subject_ids, others.subject_ids], axis=0),
        )


def _context_from_timestamps(timestamps: pd.Series) -> np.ndarray:
    """Cyclical weekday context. Purely a function of the row's own timestamp."""
    dows = np.array([ts.weekday() for ts in timestamps], dtype=np.float64)
    return np.stack([np.sin(2.0 * np.pi * dows / 7.0), np.cos(2.0 * np.pi * dows / 7.0)], axis=-1)


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 7,
) -> WindowBatch:
    """Builds causal windows grouped by subject (subjects are never mixed).

    The label of a window is the ground-truth label of its *final* observation.
    No future observation enters any window or context vector.
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, received {window_size}")

    batches: list[WindowBatch] = []
    for subject_id, group in df.groupby("subject_id", sort=False):
        g = group.sort_values(by="timestamp").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g["ground_truth_label"].to_numpy(dtype=np.int64)
        quality_col = g["quality_score"].to_numpy(dtype=np.float32) if "quality_score" in g.columns else np.ones(len(g), dtype=np.float32)
        quality = np.repeat(quality_col[:, None], len(feature_cols), axis=1)
        context = _context_from_timestamps(g["timestamp"])

        windows = PreprocessingPipeline.create_sliding_windows(feats, window_size=window_size)
        if len(windows) == 0:
            continue

        tail = window_size - 1  # index of the window-ending observation
        batches.append(
            WindowBatch(
                x=windows.astype(np.float32),
                y=labels[tail:],
                context=context[tail:].astype(np.float32),
                quality=quality[tail:].astype(np.float32),
                subject_ids=np.array([subject_id] * len(windows)),
            )
        )

    if not batches:
        raise ValueError(
            "No windows could be constructed; every subject has fewer rows than window_size "
            f"({window_size})."
        )

    total = batches[0].empty_like()
    for batch in batches:
        total = total.concat(batch)
    return total


def subject_mean_baseline(
    model_encoder,
    batch: WindowBatch,
    hidden_dim: int,
) -> dict[str, torch.Tensor]:
    """Computes one mean latent baseline memory per subject from a window batch.

    ``model_encoder`` maps ``(x_seq, context)`` to a latent vector. This encodes
    only historical (calibration) windows, so it cannot leak evaluation data.
    """
    x_t, ctx_t, _ = batch.tensors()
    with torch.no_grad():
        embeddings = model_encoder(x_t, ctx_t).numpy()

    memories: dict[str, torch.Tensor] = {}
    for subject_id in np.unique(batch.subject_ids):
        mask = batch.subject_ids == subject_id
        mean_embedding = embeddings[mask].mean(axis=0)
        memories[str(subject_id)] = torch.tensor(mean_embedding, dtype=torch.float32).reshape(1, hidden_dim)
    return memories


@dataclass
class FeatureStandardizer:
    """Population-level z-score standardisation fitted on calibration windows only.

    Two reasons this step exists:

    1. Raw physiological features span very different scales (step counts in the
       thousands, resting heart rate in the tens). Fed directly into a recurrent
       encoder they saturate its activations, so the latent representation barely
       reacts to the input and training collapses to a constant score.
    2. It is deliberately **population-level, not per-subject**. Standardising with
       each subject's own calibration statistics would encode that subject's
       personal baseline into the input itself, making any learned personal memory
       redundant and confounding every personalization ablation. Here
       normalisation only removes scale; personalisation remains the model's job.

    Statistics come from calibration windows only, so no evaluation information
    enters the transform.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, batch: WindowBatch) -> FeatureStandardizer:
        pooled = batch.x.reshape(-1, batch.x.shape[-1]).astype(np.float64)
        self.mean = pooled.mean(axis=0)
        std = pooled.std(axis=0)
        self.std = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, batch: WindowBatch) -> WindowBatch:
        """Returns a copy of ``batch`` with standardised observations."""
        if self.mean is None or self.std is None:  # pragma: no cover
            raise RuntimeError("FeatureStandardizer.transform called before fit")
        x = (batch.x.astype(np.float64) - self.mean) / self.std
        return replace(batch, x=x.astype(np.float32))


def per_window_subject_memories(
    model_encoder,
    batch: WindowBatch,
    hidden_dim: int,
    *,
    leave_one_out: bool = True,
) -> torch.Tensor:
    """Builds a personal baseline memory *for every training window*.

    Training needs the same input distribution the model sees at inference: a
    subject-specific baseline. Feeding a cohort-level baseline instead is a
    train/inference mismatch that lets the deviation module collapse to a
    constant score.

    With ``leave_one_out=True`` the memory for a window is the mean embedding of
    that subject's *other* calibration windows, so a window never contributes to
    its own baseline. The memory is detached — it is context, not a parameter.
    """
    x_t, ctx_t, _ = batch.tensors()
    with torch.no_grad():
        embeddings = model_encoder(x_t, ctx_t)

    memories = torch.zeros_like(embeddings)
    for subject_id in np.unique(batch.subject_ids):
        indices = np.where(batch.subject_ids == subject_id)[0]
        block = embeddings[indices]
        count = block.shape[0]
        if leave_one_out and count > 1:
            total = block.sum(dim=0, keepdim=True)
            memories[indices] = (total - block) / (count - 1)
        else:
            memories[indices] = block.mean(dim=0, keepdim=True)
    return memories


def expand_subject_memories(
    memories: dict[str, torch.Tensor],
    batch: WindowBatch,
    hidden_dim: int,
    default: str = "zeros",
) -> torch.Tensor:
    """Expands a per-subject memory mapping into a per-window tensor.

    Subjects absent from ``memories`` receive either a zero memory (``"zeros"``)
    or the cohort mean (``"cohort_mean"``), which is the honest cold-start
    fallback.
    """
    if default == "cohort_mean" and memories:
        stacked = torch.cat(list(memories.values()), dim=0)
        fallback = stacked.mean(dim=0, keepdim=True)
    else:
        fallback = torch.zeros(1, hidden_dim)

    rows = [memories.get(str(subject_id), fallback) for subject_id in batch.subject_ids]
    return torch.cat(rows, dim=0)
