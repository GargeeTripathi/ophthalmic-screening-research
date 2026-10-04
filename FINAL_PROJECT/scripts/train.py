"""Train ConvNeXt-Tiny on the patient-disjoint folds.

    python scripts/train.py                    # all five folds (about an hour on a laptop CPU)
    python scripts/train.py --folds 0          # only the deployed model
    python scripts/train.py --skip-completed   # resume
    python scripts/train.py --smoke            # 2-minute code-path check, writes to results/_smoke
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    a = p.parse_args()
    raise SystemExit(main(a.folds, a.smoke, a.skip_completed))
