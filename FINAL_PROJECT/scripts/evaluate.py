"""Evaluate the trained checkpoints and write every metric, table and figure.

    python scripts/evaluate.py                  # pooled out-of-fold + the deployed checkpoint
    python scripts/evaluate.py --skip-deployed  # faster: skip the file-by-file run of the served model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate import main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-deployed", action="store_true")
    raise SystemExit(main(p.parse_args().skip_deployed))
