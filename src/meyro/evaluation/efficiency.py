"""Model efficiency measurement (Phase 29).

Reports the computational cost of each architecture so a review can see the
accuracy/latency trade-off, not accuracy alone. Latency is measured
wall-clock on the current CPU; it is a like-for-like comparison across models
measured in the same run, not an absolute deployment figure.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from meyro.models.registry import count_parameters


def measure_efficiency(
    model: nn.Module,
    forward_callable: Callable[[], object],
    *,
    n_warmup: int = 3,
    n_runs: int = 30,
) -> dict[str, float]:
    """Measures parameter count, latency and throughput for one model.

    Args:
        model: the module being measured (for parameter counting).
        forward_callable: zero-argument callable performing one forward pass.
        n_warmup: untimed warm-up passes.
        n_runs: timed passes.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            forward_callable()

        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            forward_callable()
            timings.append((time.perf_counter() - start) * 1000.0)

    latencies = np.array(timings, dtype=np.float64)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    mean_latency = float(latencies.mean())

    return {
        "parameters": float(count_parameters(model)),
        "mean_latency_ms": round(mean_latency, 4),
        "std_latency_ms": round(float(latencies.std(ddof=0)), 4),
        "throughput_per_sec": round(1000.0 / mean_latency, 2) if mean_latency > 0 else 0.0,
        "model_size_kb": round(parameter_bytes / 1024.0, 2),
    }
