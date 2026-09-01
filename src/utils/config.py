"""Config loading and path resolution.

Every script starts here. Paths in configs/paths.yaml that are repo-relative get resolved
against the repo root, so a script behaves the same whether it is run from the repo root,
from scripts/, or from a Colab working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/utils/config.py -> src/utils -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_CONFIG_NAMES = ("paths", "data", "training", "idp", "experiments")


def load_config(name: str) -> dict[str, Any]:
    """Load one config file by stem, e.g. load_config("data")."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_all() -> dict[str, dict[str, Any]]:
    """Load every known config. Used by the health check and by tests."""
    return {name: load_config(name) for name in _CONFIG_NAMES}


def repo_path(*parts: str) -> Path:
    """Absolute path inside the repo."""
    return REPO_ROOT.joinpath(*parts)


def output_dir(key: str, create: bool = True) -> Path:
    """Resolve a repo-relative output directory from paths.yaml (e.g. "results")."""
    paths = load_config("paths")
    for section in ("data", "outputs"):
        if key in paths.get(section, {}):
            directory = repo_path(paths[section][key])
            if create:
                directory.mkdir(parents=True, exist_ok=True)
            return directory
    raise KeyError(f"{key!r} is not a directory key in configs/paths.yaml")


def source_path(key: str) -> Path:
    """Resolve a read-only source dataset path. Never written to."""
    paths = load_config("paths")
    if key not in paths.get("source", {}):
        raise KeyError(f"{key!r} is not a source key in configs/paths.yaml")
    return Path(paths["source"][key])
