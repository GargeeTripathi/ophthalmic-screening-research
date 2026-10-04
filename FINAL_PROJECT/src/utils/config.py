"""Config loading and path resolution. Every script starts here.

configs/config.yaml is the single source of truth. Machine-specific locations (where the ODIR
images live) come from the ODIR_IMAGES environment variable or a project-root .env file, so no
tracked file contains an absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_dotenv() -> None:
    """Read project-root .env into the environment. Real environment variables win."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def load_config(name: str = "config") -> dict[str, Any]:
    """Load one config file by stem: load_config() or load_config("idp")."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def repo_path(*parts: str) -> Path:
    """Absolute path inside the project."""
    return REPO_ROOT.joinpath(*parts)


def images_dir() -> Path | None:
    """Folder with the ODIR JPEGs, or None if it is not configured or does not exist."""
    value = os.environ.get("ODIR_IMAGES") or load_config()["data"].get("images_dir") or ""
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.is_dir() else None
