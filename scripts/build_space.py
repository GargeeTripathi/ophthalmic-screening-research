"""Assemble the Hugging Face Space that runs the full prototype.

    python scripts/build_space.py [--out DIR]

The Space runs scripts/serve.py unchanged. That is the entire point of deploying it this
way: the document pathway is OCR plus layout-aware extraction plus a learned reliability
estimator, all Python, and a static host cannot run any of it. Rather than reimplement that
in JavaScript and then be unable to prove the reimplementation agrees, the real code is
shipped and run.

WHAT IS DELIBERATELY LEFT OUT
-----------------------------
data/, results/, figures/, reports/, docs/ and the two large checkpoints. The Space needs
what serves a case, not what reproduces the study: the 34 KB head, the reliability estimator
and the deployment manifest. The ImageNet backbone is fetched from torchvision at image
build time, exactly as a fresh clone does.

ODIR images are not redistributed, so the deployed instance starts upload-only.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import REPO_ROOT  # noqa: E402

# Only what scripts/serve.py imports or reads at runtime.
TREES = ("configs", "src")
FILES = ("scripts/serve.py",
         "checkpoints/screening_head.pt",
         "checkpoints/reliability.joblib",
         "checkpoints/deployment.json")
SPACE_FILES = ("Dockerfile", "requirements-space.txt", "README.md")

# Caches and test noise that shutil.copytree would otherwise carry into the image.
JUNK = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "build" / "space")
    args = parser.parse_args()
    out: Path = args.out

    print("=" * 72)
    print("BUILD HUGGING FACE SPACE")
    print("=" * 72)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for tree in TREES:
        shutil.copytree(REPO_ROOT / tree, out / tree, ignore=JUNK)
        print(f"  copied {tree}/")

    missing = []
    for name in FILES:
        source = REPO_ROOT / name
        if not source.is_file():
            missing.append(name)
            continue
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
        print(f"  copied {name}  ({source.stat().st_size / 1024:.0f} KB)")

    if missing:
        print("\nMISSING, and the Space cannot start without them:")
        for name in missing:
            print(f"  {name}")
        print("run scripts/export_model.py first")
        return 2

    for name in SPACE_FILES:
        shutil.copy(REPO_ROOT / "deploy" / "space" / name, out / name)
        print(f"  copied {name}")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n{out}  ({total / 1e6:.1f} MB, {sum(1 for _ in out.rglob('*') if _.is_file())} files)")
    print("\nnext:")
    print("  huggingface-cli login")
    print(f"  huggingface-cli upload <user>/ophthalmic-screening {out} . --repo-type=space")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
