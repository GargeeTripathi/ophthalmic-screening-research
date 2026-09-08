"""Rebuild the entire project from the source datasets, in one command.

Every stage declares what it produces, so a stage whose outputs already exist is skipped.
That makes this safe to re-run: it resumes rather than restarting, and a crash costs only
the stage that was running.

    python scripts/run_all.py               # run everything not already done
    python scripts/run_all.py --force       # ignore caches and redo everything
    python scripts/run_all.py --from E7     # start from a named stage
    python scripts/run_all.py --list        # show the plan and what is already cached
    python scripts/run_all.py --dry-run

Total from cold on a 14-thread CPU: roughly 70 minutes, dominated by fingerprints (~11 min),
embeddings (~18 min) and OCR (~35 min). Everything else is minutes.

The GPU half - E2/E3 fine-tuned baselines - is deliberately NOT here. It runs in
colab/train_models.ipynb and needs a Drive upload; see README.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# (stage id, script, human description, output files that mean "already done")
STAGES: list[tuple[str, str, str, list[str]]] = [
    ("manifest", "build_manifest.py", "ODIR manifest, expected counts asserted",
     ["data/manifests/odir_manifest.csv"]),
    ("merged", "build_merged_manifest.py", "provenance labels for the audit subject",
     ["data/manifests/merged_manifest.csv"]),
    ("fingerprints", "build_fingerprints.py", "FOV-normalised fingerprints (~11 min)",
     ["data/processed/fingerprints_odir.npz", "data/processed/fingerprints_merged.npz"]),
    ("embeddings", "build_embeddings.py", "ResNet50 embeddings (~18 min)",
     ["data/processed/embeddings_odir.npz", "data/processed/embeddings_merged.npz"]),
    ("dedup", "find_duplicates.py", "duplicate detection, calibrated on free ground truth",
     ["data/manifests/odir_manifest_clean.csv", "data/manifests/duplicate_groups.csv"]),
    ("splits", "build_splits.py", "four protocol rungs, patient-disjointness asserted",
     ["data/splits/splits_patient_disjoint_dedup.csv"]),
    ("E0", "provenance_probe.py", "E0a/E0b/E0c provenance probes",
     ["results/E0_provenance_probes.json"]),
    ("E1", "protocol_ladder.py", "E1 protocol ladder",
     ["results/E1_protocol_ladder.json"]),
    ("E1b", "capacity_ladder.py", "E1b capacity ladder - does inflation need memorisation?",
     ["results/E1b_capacity_ladder.json"]),
    ("E5", "cross_source_transfer.py", "E5 transfer to unseen acquisition sources",
     ["results/E5_cross_source_transfer.json"]),
    ("E11", "confound_sensitivity.py", "E11 age/sex controls",
     ["results/E11_confound_sensitivity.json"]),
    ("documents", "generate_documents.py", "360 synthetic referrals, layouts D-F sealed",
     ["data/documents/ground_truth_dev.json", "data/documents/ground_truth_sealed.json"]),
    ("ocr", "run_ocr.py", "OCR the corpus (~35 min)",
     ["data/processed/ocr_dev.json", "data/processed/ocr_sealed.json"]),
    ("E7", "evaluate_extraction.py", "E7 extraction benchmark + E8 reliability",
     ["results/E7_extraction.json", "results/E8_reliability.json"]),
    ("E9", "run_deferral.py", "E9 deferral + E10 robustness matrix",
     ["results/E9_deferral.json", "results/E10_robustness_matrix.csv"]),
    ("E12", "run_xai.py", "E12 explainability battery",
     ["results/E12_xai.json"]),
    ("E13", "statistical_validation.py", "E13 seed stability + Holm-Bonferroni",
     ["results/E13_statistical_validation.json"]),
    ("figures", "make_figures.py", "all figures", []),          # always regenerate
    ("report", "summarise_results.py", "reports/RESULTS.md", []),
    ("worklist", "build_worklist.py", "reports/worklist_report.html", []),
]


def is_cached(outputs: list[str]) -> bool:
    return bool(outputs) and all((REPO / path).exists() for path in outputs)


def run(script: str) -> tuple[bool, float]:
    started = time.time()
    result = subprocess.run([PYTHON, str(REPO / "scripts" / script)], cwd=REPO)
    return result.returncode == 0, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="ignore caches")
    parser.add_argument("--from", dest="start", help="start from this stage id")
    parser.add_argument("--list", action="store_true", help="show the plan and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"{'stage':<14}{'script':<30}{'cached':>8}  description")
        for stage, script, description, outputs in STAGES:
            mark = "yes" if is_cached(outputs) else ("always" if not outputs else "no")
            print(f"{stage:<14}{script:<30}{mark:>8}  {description}")
        return 0

    plan = STAGES
    if args.start:
        ids = [s[0] for s in STAGES]
        if args.start not in ids:
            print(f"unknown stage {args.start!r}; choose from {', '.join(ids)}")
            return 2
        plan = STAGES[ids.index(args.start):]

    print("=" * 72)
    print("RUN ALL")
    print("=" * 72)

    timings, skipped, failed = [], [], []
    for stage, script, description, outputs in plan:
        if not args.force and is_cached(outputs):
            print(f"\n[skip] {stage:<12} cached  ({description})")
            skipped.append(stage)
            continue
        print(f"\n[run ] {stage:<12} {description}")
        if args.dry_run:
            continue
        ok, elapsed = run(script)
        timings.append((stage, elapsed))
        if ok:
            print(f"[done] {stage:<12} {elapsed / 60:.1f} min")
        else:
            print(f"[FAIL] {stage:<12} after {elapsed / 60:.1f} min - stopping")
            failed.append(stage)
            break

    print("\n" + "=" * 72)
    if timings:
        total = sum(t for _, t in timings)
        print(f"ran {len(timings)} stage(s) in {total / 60:.1f} min")
        for stage, elapsed in sorted(timings, key=lambda kv: -kv[1])[:5]:
            print(f"  {stage:<14}{elapsed / 60:>6.1f} min")
    if skipped:
        print(f"skipped {len(skipped)} cached stage(s): {', '.join(skipped)}")
    if failed:
        print(f"\nRESULT: FAILED at {failed[0]}")
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
