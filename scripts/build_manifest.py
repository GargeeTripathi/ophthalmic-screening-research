"""Build the canonical ODIR image manifest and assert it matches the expected dataset.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --no-sizes    # skip image header reads (faster)

Exit codes:
    0  manifest written and matches configs/data.yaml
    2  manifest written but the counts do NOT match - investigate before proceeding
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import (  # noqa: E402
    build_manifest,
    check_expected,
    summarise,
    write_manifest,
)
from src.utils.config import output_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sizes", action="store_true",
                        help="skip reading image dimensions (width/height stay empty)")
    args = parser.parse_args()

    print("=" * 72)
    print("BUILD MANIFEST")
    print("=" * 72)

    print("\nReading ODIR label table and resolving image paths...")
    manifest = build_manifest(read_sizes=not args.no_sizes)

    skipped = manifest.attrs.get("skipped", {})
    print("\nRows skipped while building:")
    for reason, count in skipped.items():
        print(f"  {reason:<16} {count:>6}")

    stats = summarise(manifest)

    print("\n" + "-" * 72)
    print("CLASS DISTRIBUTION")
    print("-" * 72)
    print(stats["by_class"].to_string())

    print("\n" + "-" * 72)
    print("STRUCTURE")
    print("-" * 72)
    print(f"  images                      {stats['total_images']}")
    print(f"  patients                    {stats['total_patients']}")
    print(f"  eyes                        {stats['eye_counts']}")
    print(f"  patients with both eyes     {stats['patients_with_both_eyes']}")
    print(f"  patients with one eye       {stats['patients_with_one_eye']}")
    print(f"  patients spanning 2 classes {stats['patients_spanning_two_classes']}")
    print(f"  missing age / sex           {stats['missing_age']} / {stats['missing_sex']}")
    print(f"  unreadable images           {stats['unreadable_images']}")
    print(f"  distinct resolutions        {stats['distinct_resolutions']}")

    print("\n" + "-" * 72)
    print("EXPECTED-COUNT ASSERTIONS")
    print("-" * 72)
    problems = check_expected(manifest)
    if problems:
        for problem in problems:
            print(f"  MISMATCH  {problem}")
    else:
        print("  all counts match configs/data.yaml")

    path = write_manifest(manifest)

    summary = {k: v for k, v in stats.items() if k != "by_class"}
    summary["by_class"] = stats["by_class"].to_dict(orient="index")
    summary["skipped"] = skipped
    summary["mismatches"] = problems
    summary_path = output_dir("manifests") / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"WRITTEN\n  {path}\n  {summary_path}")
    if problems:
        print("\nRESULT: MISMATCH - the source data is not what the config expects.")
        print("Do not build splits or run experiments until this is understood.")
        return 2
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
