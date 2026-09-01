"""Project health check. Run this before anything else, and after any environment change.

Verifies, in order: Python version, required packages, repo directory structure, config
files parse and hold their required keys, and the read-only source datasets are reachable
with the file counts the audit expects.

Exit code 0 = healthy, 1 = something is wrong. Non-zero exit is deliberate so this can
gate a CI step or a Colab cell later.

    python scripts/health_check.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import REPO_ROOT, load_all, source_path  # noqa: E402

MIN_PYTHON = (3, 10)

# import name -> what it is for, so a missing package explains itself
REQUIRED = {
    "yaml": "config loading",
    "numpy": "arrays",
    "pandas": "manifests and results tables",
    "PIL": "image inspection",
    "sklearn": "metrics, probes, calibration",
    "matplotlib": "figures",
    "scipy": "statistics",
}
# Needed later, absent is fine during bootstrap
OPTIONAL = {
    "torch": "embeddings and GPU training",
    "torchvision": "backbones",
    "cv2": "image utilities",
    # RapidOCR replaced Tesseract - pip-installable, no system binary, no admin rights.
    "rapidocr_onnxruntime": "OCR",
    "rapidfuzz": "fuzzy label matching in extraction",
}

EXPECTED_DIRS = [
    "configs", "src", "scripts", "tests", "notebooks", "colab",
    "data/raw", "data/manifests", "data/processed", "data/documents", "data/splits",
    "checkpoints", "results", "figures", "reports",
]

REQUIRED_CONFIG_KEYS = {
    "paths": ["source", "data", "outputs", "colab"],
    "data": ["classes", "expected", "manifest_columns", "splits"],
    "training": ["seed", "backbones", "optimizer", "calibration"],
    "idp": ["layouts", "fields", "degradation", "diagnosis_corruption"],
    "experiments": ["E0b", "E1", "statistics"],
}

# source key -> (expected file count, None means "exists is enough")
SOURCE_CHECKS = {
    "odir_label_csv": None,
    "odir_images_train": 7000,
    "merged_root": None,
}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  [ ok ] {label}{'  ' + detail if detail else ''}")

    def warn(self, label: str, detail: str) -> None:
        self.warnings.append(label)
        print(f"  [warn] {label}  {detail}")

    def fail(self, label: str, detail: str) -> None:
        self.failures.append(label)
        print(f"  [FAIL] {label}  {detail}")


def check_python(report: Report) -> None:
    print("\nPython")
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version[:2] >= MIN_PYTHON:
        report.ok("interpreter", f"{text}  ({sys.executable})")
    else:
        report.fail("interpreter", f"{text} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")


def check_packages(report: Report) -> None:
    print("\nPackages")
    for module, purpose in REQUIRED.items():
        try:
            loaded = importlib.import_module(module)
            report.ok(module, getattr(loaded, "__version__", ""))
        except ImportError:
            report.fail(module, f"missing - needed for {purpose}")
    for module, purpose in OPTIONAL.items():
        try:
            loaded = importlib.import_module(module)
            report.ok(module, getattr(loaded, "__version__", ""))
        except ImportError:
            report.warn(module, f"not installed - needed for {purpose}")


def check_dirs(report: Report) -> None:
    print("\nDirectories")
    missing = [d for d in EXPECTED_DIRS if not (REPO_ROOT / d).is_dir()]
    if missing:
        report.fail("structure", f"missing: {', '.join(missing)}")
    else:
        report.ok("structure", f"{len(EXPECTED_DIRS)} directories present")


def check_configs(report: Report) -> None:
    print("\nConfigs")
    try:
        configs = load_all()
    except Exception as exc:  # noqa: BLE001 - report it, do not raise
        report.fail("parse", f"{type(exc).__name__}: {exc}")
        return
    for name, keys in REQUIRED_CONFIG_KEYS.items():
        absent = [k for k in keys if k not in configs.get(name, {})]
        if absent:
            report.fail(f"{name}.yaml", f"missing keys: {', '.join(absent)}")
        else:
            report.ok(f"{name}.yaml", f"{len(configs[name])} top-level keys")

    # Invariant worth locking on day one: dev and sealed IDP layouts must never overlap.
    layouts = configs.get("idp", {}).get("layouts", {})
    overlap = set(layouts.get("dev", [])) & set(layouts.get("sealed", []))
    if overlap:
        report.fail("idp layouts", f"dev and sealed overlap: {sorted(overlap)}")
    else:
        report.ok("idp layouts", f"dev {layouts.get('dev')} / sealed {layouts.get('sealed')} disjoint")


def check_sources(report: Report) -> None:
    print("\nSource data (read-only)")
    for key, expected_count in SOURCE_CHECKS.items():
        try:
            path = source_path(key)
        except KeyError as exc:
            report.fail(key, str(exc))
            continue
        if not path.exists():
            report.fail(key, f"not found: {path}")
            continue
        if expected_count is None:
            report.ok(key, str(path))
            continue
        count = sum(1 for entry in path.iterdir() if entry.is_file())
        if count == expected_count:
            report.ok(key, f"{count} files")
        else:
            report.warn(key, f"{count} files, expected {expected_count}")


def main() -> int:
    print("=" * 72)
    print("PROJECT HEALTH CHECK")
    print(f"repo: {REPO_ROOT}")
    print("=" * 72)

    report = Report()
    check_python(report)
    check_packages(report)
    check_dirs(report)
    check_configs(report)
    check_sources(report)

    print("\n" + "=" * 72)
    if report.failures:
        print(f"UNHEALTHY - {len(report.failures)} failure(s): {', '.join(report.failures)}")
        if report.warnings:
            print(f"plus {len(report.warnings)} warning(s)")
        return 1
    if report.warnings:
        print(f"HEALTHY - with {len(report.warnings)} warning(s): {', '.join(report.warnings)}")
    else:
        print("HEALTHY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
