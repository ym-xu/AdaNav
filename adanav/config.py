"""Dataset paths and build settings, read from configs/datasets.yaml.

configs/datasets.local.yaml (git-ignored) takes precedence when it exists, so
machine-specific paths stay out of the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "datasets.local.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        path = LOCAL_CONFIG if LOCAL_CONFIG.exists() else DEFAULT_CONFIG
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dataset_paths(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a dataset entry; relative paths are taken from the repo root."""
    if name not in config or name == "split":
        known = ", ".join(k for k in config if k != "split")
        raise SystemExit(f"unknown dataset '{name}' (known: {known})")
    entry = dict(config[name])
    for key in ("mineru_root", "doctree_dir"):
        p = Path(entry[key])
        entry[key] = p if p.is_absolute() else REPO_ROOT / p
    return entry
