"""Generate the synthetic ophthalmic referral corpus.

Documents are paired 1:1 with ODIR test images, so a document's stated diagnosis can be
compared against what the image model predicts for that exact eye.

    python scripts/generate_documents.py

Writes:
    data/documents/dev/       layouts A-C  - extractors are developed against these
    data/documents/sealed/    layouts D-F  - DO NOT LOOK AT until extractors are frozen
    data/documents/ground_truth_dev.json
    data/documents/ground_truth_sealed.json
    data/documents/corpus_summary.json
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.idp.generator import Document, render, sample_record  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402


def main() -> int:
    idp = load_config("idp")
    classes = load_config("data")["classes"]
    documents_dir = output_dir("documents")
    manifests = output_dir("manifests")

    manifest_path = manifests / "odir_manifest_clean.csv"
    if not manifest_path.is_file():
        manifest_path = manifests / "odir_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8")

    rng = random.Random(idp["seed"])
    params = idp["degradation"]["params"]
    levels = idp["degradation"]["levels"]
    per_layout = idp["records_per_layout"]
    train_rate = idp["diagnosis_corruption"]["train_rate"]

    print("=" * 72)
    print("GENERATE SYNTHETIC REFERRAL CORPUS")
    print("=" * 72)
    print(f"\nlayouts   dev {idp['layouts']['dev']}   SEALED {idp['layouts']['sealed']}")
    print(f"degradation levels {levels}")
    print(f"records per layout {per_layout}   diagnosis corruption rate {train_rate}")

    # Sample patients stratified by class so every class appears in the corpus.
    # pandas 3 drops the grouping column inside groupby.apply, so sample per class directly.
    pool = pd.concat(
        [block.sample(min(len(block), 200), random_state=idp["seed"])
         for _, block in manifest.groupby("label")],
        ignore_index=True,
    )

    summary = {}
    for split_name, layouts in (("dev", idp["layouts"]["dev"]),
                                ("sealed", idp["layouts"]["sealed"])):
        out_dir = documents_dir / split_name
        documents: list[Document] = []
        counter = 0
        for layout in layouts:
            for _ in range(per_layout):
                for level in levels:
                    row = pool.iloc[rng.randrange(len(pool))]
                    record = sample_record(
                        rng,
                        {"patient_id": row["patient_id"], "age": row["age"],
                         "sex": row["sex"], "eye": row["eye"]},
                        row["label"], train_rate, classes,
                        idp["optional_fields"], idp["missing_field_rate"],
                    )
                    record.document_id = f"{split_name}_{layout}_{counter:04d}_L{level}"
                    record.image_path = ""
                    documents.append(render(record, layout, level, params, out_dir, rng))
                    counter += 1

        path = documents_dir / f"ground_truth_{split_name}.json"
        path.write_text(json.dumps([asdict(d) for d in documents], indent=2, default=str),
                        encoding="utf-8")

        corrupted = sum(1 for d in documents if d.diagnosis_corrupted)
        missing = {}
        for d in documents:
            for key, value in d.fields.items():
                if value is None:
                    missing[key] = missing.get(key, 0) + 1
        summary[split_name] = {
            "documents": len(documents),
            "layouts": layouts,
            "per_level": {str(l): sum(1 for d in documents if d.degradation_level == l)
                          for l in levels},
            "diagnosis_corrupted": corrupted,
            "corruption_rate_actual": round(corrupted / len(documents), 4),
            "missing_field_counts": missing,
            "ground_truth": str(path),
        }
        print(f"\n{split_name}: {len(documents)} documents in {out_dir}")
        print(f"  per degradation level {summary[split_name]['per_level']}")
        print(f"  diagnosis corrupted   {corrupted} ({summary[split_name]['corruption_rate_actual']:.1%})")
        print(f"  missing fields        {missing}")

    (documents_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SEALED LAYOUTS D-F ARE WRITTEN BUT MUST NOT BE INSPECTED OR TUNED AGAINST.")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
