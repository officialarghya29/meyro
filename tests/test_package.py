"""Phase 0 smoke tests.

These verify only what exists at initialization: that the package imports, that
its metadata is well-formed, and that the repository skeleton matches the
master specification. No modeling behaviour is tested here because none exists
yet — later phases add their own tests alongside their code.
"""

from __future__ import annotations

from pathlib import Path

import meyro

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "docs",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "src/meyro",
    "experiments",
    "notebooks",
    "tests",
    "scripts",
    "api",
    "frontend",
    "deployment",
    "models",
]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CITATION.cff",
    ".gitignore",
    ".env.example",
    "requirements.txt",
    "requirements-dev.txt",
]


def test_package_imports() -> None:
    assert meyro.__version__ == "0.0.1"


def test_tagline_is_canonical() -> None:
    """The tagline is product identity; it must not drift."""
    assert meyro.TAGLINE == "AI that learns your normal."


def test_version_is_pep440_shaped() -> None:
    parts = meyro.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_required_directories_exist() -> None:
    missing = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_required_files_exist() -> None:
    missing = [f for f in REQUIRED_FILES if not (REPO_ROOT / f).is_file()]
    assert not missing, f"missing files: {missing}"


def test_no_env_file_is_committed() -> None:
    """Rule 21: secrets stay out of Git. `.env` must never be tracked."""
    assert not (REPO_ROOT / ".env").exists()


def test_data_dirs_are_placeholders_only() -> None:
    """Rule 22: no health/personal data in the repository."""
    for sub in ("raw", "interim", "processed"):
        contents = {p.name for p in (REPO_ROOT / "data" / sub).iterdir()}
        assert contents <= {".gitkeep"}, f"data/{sub} contains non-placeholder files: {contents}"
