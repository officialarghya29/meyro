"""Model version registry and checkpoint persistence (Phase 30).

Every checkpoint is written together with the configuration, seed, dataset
descriptor and git commit that produced it, so a saved model is never ambiguous
about its provenance. Version names follow the workflow: ``MEYRO-V1``,
``MEYRO-V2``, ... Checkpoints live under ``models/`` (git-ignored binary blobs).

IMPORTANT: this module records provenance metadata only. It does not fabricate
metrics; callers must pass metrics measured elsewhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from meyro.utils.io import environment_fingerprint, write_json

MODELS_DIR = Path("models")


@dataclass
class ModelCardMetadata:
    """Provenance record stored alongside every checkpoint."""

    model_name: str
    version: str
    architecture: str
    config: dict[str, Any] = field(default_factory=dict)
    dataset_descriptor: str = ""
    n_parameters: int = 0
    seed: int = 42
    training_epochs: int = 0
    training_windows: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_checkpoint(
    model: nn.Module,
    metadata: ModelCardMetadata,
    *,
    directory: Path | str = MODELS_DIR,
) -> Path:
    """Persists model weights plus a JSON provenance sidecar.

    Returns the path of the written ``.pt`` checkpoint; the sidecar is written
    next to it as ``<version>.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    metadata.n_parameters = count_parameters(model)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": asdict(metadata),
        "environment": environment_fingerprint(),
    }

    checkpoint_path = directory / f"{metadata.version}.pt"
    torch.save(payload, checkpoint_path)
    write_json(directory / f"{metadata.version}.json", payload["metadata"] | {"environment": payload["environment"]})
    return checkpoint_path


def load_checkpoint(
    version: str,
    model: nn.Module,
    *,
    directory: Path | str = MODELS_DIR,
    strict: bool = True,
) -> ModelCardMetadata:
    """Loads weights into ``model`` and returns the stored metadata."""
    directory = Path(directory)
    payload = torch.load(directory / f"{version}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=strict)
    return ModelCardMetadata(**payload["metadata"])
