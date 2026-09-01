"""OCR the document corpus and cache the tokens.

Slow, so it is cached and never recomputed. Run once per corpus regeneration.

    python scripts/run_ocr.py
    python scripts/run_ocr.py --split sealed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.idp.ocr import read, save  # noqa: E402
from src.utils.config import output_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "sealed", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    documents_dir = output_dir("documents")
    processed = output_dir("processed")
    splits = ["dev", "sealed"] if args.split == "both" else [args.split]

    print("=" * 72)
    print("RUN OCR  (RapidOCR / ONNX Runtime, CPU)")
    print("=" * 72)

    for split in splits:
        cache = processed / f"ocr_{split}.json"
        if cache.is_file() and not args.force:
            print(f"\n{split}: cached - skipping")
            continue

        truth_path = documents_dir / f"ground_truth_{split}.json"
        records = json.loads(truth_path.read_text(encoding="utf-8"))
        print(f"\n{split}: {len(records)} documents")

        started = time.time()
        tokens_by_document = {}
        for position, record in enumerate(records, start=1):
            tokens_by_document[record["document_id"]] = read(record["image_path"])
            if position % 30 == 0:
                print(f"    {position}/{len(records)}")
        elapsed = time.time() - started

        save(cache, tokens_by_document)
        empty = sum(1 for t in tokens_by_document.values() if not t)
        mean_tokens = sum(len(t) for t in tokens_by_document.values()) / len(records)
        print(f"  done in {elapsed / 60:.1f} min   mean {mean_tokens:.1f} tokens/doc"
              f"   empty results: {empty}")
        print(f"  written: {cache}")

    print("\n" + "=" * 72)
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
