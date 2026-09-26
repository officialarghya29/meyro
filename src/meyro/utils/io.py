"""Deterministic seeding and machine-readable artifact helpers (Phase 25/30).

Master workflow rule 7 (deterministic seeds) and rule 8 (experiment tracking):
every experiment must be reproducible and every result must be written to a
machine-readable artifact rather than transcribed by hand.
"""

from __future__ import annotations

import contextlib
import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Committed artifact locations (``results/`` is git-ignored generated output).
EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")


def set_global_seed(seed: int = 42) -> None:
    """Seeds Python, NumPy and PyTorch, and forces deterministic execution.

    Seeding alone is not enough for reproducibility. Multi-threaded recurrent
    kernels on CPU reduce in a nondeterministic order, which made results vary
    between identical runs — for one drift experiment the same configuration
    produced a 2% and a 93% persistent-false-alarm rate on separate runs.

    Threading is therefore pinned to a single thread and deterministic kernels
    are requested. This costs wall-clock time; correctness of the reported
    numbers takes priority.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        # Not all kernels implement deterministic variants; warn_only avoids a
        # hard failure on older torch versions.
        with contextlib.suppress(RuntimeError, AttributeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pass


def git_commit_short() -> str:
    """Returns the short HEAD commit hash, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def environment_fingerprint() -> dict[str, str]:
    """Captures the hardware/software context required by rule 8."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit_short(),
    }


def write_json(path: str | Path, payload: Any) -> Path:
    """Writes ``payload`` as pretty JSON, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return target


def save_experiment_results(
    suite: str,
    payload: dict[str, Any],
    *,
    seed: int | None = None,
) -> Path:
    """Persists a suite's results into the committed ``experiments/`` tree.

    Follows master workflow rule 8 by recording the configuration, seed,
    environment and timestamp alongside the metrics themselves.
    """
    enveloped = {
        "suite": suite,
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": seed,
        "environment": environment_fingerprint(),
        "results": payload,
    }
    return write_json(EXPERIMENTS_DIR / suite / "results.json", enveloped)
