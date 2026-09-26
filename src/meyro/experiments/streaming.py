"""Streaming (online) scoring for the MEYRO family.

A static one-shot evaluation cannot exercise adaptation: at the first scored
step the fast and slow memories are identical by construction, so ablating the
dual-timescale mechanism changes nothing. MEYRO is a *longitudinal* model, so it
must be scored the way it runs in production — window by window, in time order,
with the baseline memory evolving causally from observations already seen.

For every subject:

1. score the current window against the memory built from *previous* windows;
2. update the memory using the model's own gated rule;
3. proceed to the next window.

Scores are returned in the batch's original window order, so they align exactly
with ``batch.y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from meyro.data.windows import WindowBatch
from meyro.models.meyro import MEYROModel, MEYROModelV2


@dataclass
class ScoringOptions:
    """Ablation switches for streaming evaluation.

    Every option defaults to the full model, so ``ScoringOptions()`` reproduces
    the unmodified architecture.
    """

    zero_memory: bool = False
    single_timescale: bool = False
    zero_context: bool = False
    unit_quality: bool = False
    euclidean_deviation: bool = False
    persistence_gating: bool = False


def _subject_indices(batch: WindowBatch) -> list[tuple[str, np.ndarray]]:
    """Returns (subject_id, window indices) preserving batch order."""
    blocks: list[tuple[str, np.ndarray]] = []
    seen: list[str] = []
    for subject_id in batch.subject_ids:
        subject = str(subject_id)
        if subject not in seen:
            seen.append(subject)
    for subject in seen:
        blocks.append((subject, np.where(batch.subject_ids == subject)[0]))
    return blocks


def _resolve_memory(
    memories: dict[str, torch.Tensor] | None,
    subject: str,
    hidden_dim: int,
) -> torch.Tensor:
    """Returns the subject's memory, or the cohort mean when unseen (cold start)."""
    if memories and subject in memories:
        return memories[subject].clone()
    if memories:
        return torch.cat(list(memories.values()), dim=0).mean(dim=0, keepdim=True)
    return torch.zeros(1, hidden_dim)


def stream_scores_v2(
    model: MEYROModelV2,
    batch: WindowBatch,
    fast_memories: dict[str, torch.Tensor] | None,
    slow_memories: dict[str, torch.Tensor] | None,
    options: ScoringOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Streams MEYRO-V2 over a batch; returns ``(anomalies, persistences)``."""
    options = options or ScoringOptions()
    model.eval()

    anomalies = np.zeros(len(batch), dtype=np.float64)
    persistences = np.zeros(len(batch), dtype=np.float64)
    hidden_dim = model.hidden_dim

    with torch.no_grad():
        for subject, indices in _subject_indices(batch):
            fast = _resolve_memory(fast_memories, subject, hidden_dim)
            slow = _resolve_memory(slow_memories, subject, hidden_dim)
            if options.zero_memory:
                fast = torch.zeros(1, hidden_dim)
                slow = torch.zeros(1, hidden_dim)

            for window_index in indices:
                window = torch.tensor(batch.x[window_index : window_index + 1], dtype=torch.float32)
                context = torch.tensor(batch.context[window_index : window_index + 1], dtype=torch.float32)
                quality = torch.tensor(batch.quality[window_index : window_index + 1], dtype=torch.float32)

                if options.zero_context:
                    context = torch.zeros_like(context)
                if options.unit_quality:
                    quality = torch.ones_like(quality)

                slow_in = slow.clone()
                fast_in = slow_in.clone() if options.single_timescale else fast.clone()

                anomaly, _unc, persistence, _d, fast_next, slow_next = model(
                    window, context, slow_in, fast_in, quality
                )

                if options.euclidean_deviation:
                    embedding = model.encode_observation(window, context)
                    score = float(torch.norm(embedding - slow_in, p=2, dim=-1).squeeze())
                else:
                    score = float(anomaly.squeeze())
                    if options.persistence_gating:
                        score *= float(persistence.squeeze())

                anomalies[window_index] = score
                persistences[window_index] = float(persistence.squeeze())

                # Causal memory update; an ablated memory stays at zero.
                if not options.zero_memory:
                    fast = fast_next.clone()
                    slow = slow_next.clone()

    return anomalies, persistences


def stream_scores_v1(
    model: MEYROModel,
    batch: WindowBatch,
    memories: dict[str, torch.Tensor] | None,
    options: ScoringOptions | None = None,
) -> np.ndarray:
    """Streams MEYRO-V1 over a batch; returns anomalies."""
    options = options or ScoringOptions()
    model.eval()

    anomalies = np.zeros(len(batch), dtype=np.float64)
    hidden_dim = model.hidden_dim

    with torch.no_grad():
        for subject, indices in _subject_indices(batch):
            memory = _resolve_memory(memories, subject, hidden_dim)
            if options.zero_memory:
                memory = torch.zeros(1, hidden_dim)

            for window_index in indices:
                window = torch.tensor(batch.x[window_index : window_index + 1], dtype=torch.float32)
                context = torch.tensor(batch.context[window_index : window_index + 1], dtype=torch.float32)
                quality = torch.tensor(batch.quality[window_index : window_index + 1], dtype=torch.float32)

                if options.zero_context:
                    context = torch.zeros_like(context)
                if options.unit_quality:
                    quality = torch.ones_like(quality)

                anomaly, _unc, _d, memory_next = model(window, context, memory, quality)
                anomalies[window_index] = float(anomaly.squeeze())

                if not options.zero_memory:
                    memory = memory_next.clone()

    return anomalies
