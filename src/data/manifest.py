"""Build the canonical image manifest from the raw ODIR-5K label table.

One row per image. Every later stage - splits, provenance probes, training, evaluation -
reads this file and never the raw CSV, so the dataset has exactly one definition.

Three things here are worth knowing before reading the code:

1. ODIR's `labels` column is a FLATTENING. It gives one code per row, but the underlying
   diagnostic keyword often names several findings ("glaucoma, diabetic retinopathy").
   We keep the flattened label because the four-class task needs it, and we record a
   `multi_finding` flag so the sensitivity analysis can exclude those rows later.

2. Keywords are per-eye. Row `1234_left.jpg` must read `Left-Diagnostic Keywords`, not the
   right one. Getting this backwards silently mislabels half the dataset.

3. Keywords contain full-width commas from the original Chinese annotation, so every read
   and write is explicitly UTF-8.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils.config import load_config, output_dir, source_path

Image.MAX_IMAGE_PIXELS = None

# ODIR filenames are "<patient id>_<eye>.jpg"
FILENAME_PATTERN = re.compile(r"^(\d+)_(left|right)\.(jpg|jpeg|png)$", re.IGNORECASE)

# The annotation uses a full-width comma to separate multiple findings.
FINDING_SEPARATORS = ("，", ",")


def parse_filename(filename: str) -> tuple[str | None, str | None]:
    """"1234_left.jpg" -> ("1234", "left"). Returns (None, None) if it does not match."""
    match = FILENAME_PATTERN.match(str(filename))
    if not match:
        return None, None
    return match.group(1), match.group(2).lower()


def single_label(raw: str) -> str | None:
    """Extract the one class code from ODIR's stringified list, e.g. "['N']" -> "N"."""
    try:
        codes = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return codes[0] if isinstance(codes, list) and len(codes) == 1 else None


def keyword_for_eye(row: pd.Series, eye: str) -> str:
    column = "Left-Diagnostic Keywords" if eye == "left" else "Right-Diagnostic Keywords"
    return str(row.get(column, "")).strip()


def has_multiple_findings(keyword: str) -> bool:
    return any(separator in keyword for separator in FINDING_SEPARATORS)


def read_image_size(path: Path) -> tuple[int | None, int | None]:
    """Header-only read - Pillow is lazy, so this does not decode pixels."""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - a bad file becomes a null row, not a crash
        return None, None


def build_manifest(read_sizes: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Return the manifest as a DataFrame. Does not write anything."""
    data_config = load_config("data")
    keep_classes = set(data_config["classes"])

    label_csv = source_path("odir_label_csv")
    image_dir = source_path("odir_images_train")
    table = pd.read_csv(label_csv, encoding="utf-8")

    rows: list[dict] = []
    skipped = {"multi_label": 0, "dropped_code": 0, "bad_filename": 0, "missing_image": 0}

    for _, record in table.iterrows():
        code = single_label(record["labels"])
        if code is None:
            skipped["multi_label"] += 1
            continue
        if code not in keep_classes:
            skipped["dropped_code"] += 1
            continue

        filename = str(record["filename"])
        patient_id, eye = parse_filename(filename)
        if patient_id is None:
            skipped["bad_filename"] += 1
            continue

        image_path = image_dir / filename
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue

        keyword = keyword_for_eye(record, eye)
        rows.append({
            "image_id": f"odir_{patient_id}_{eye}",
            "image_path": str(image_path),
            "patient_id": f"odir_{patient_id}",
            "eye": eye,
            "label": code,
            "age": record.get("Patient Age"),
            "sex": record.get("Patient Sex"),
            "keyword": keyword,
            "source": "ODIR-5K",
            "original_filename": filename,
            "width": None,
            "height": None,
            "file_extension": Path(filename).suffix.lower(),
            "multi_finding": has_multiple_findings(keyword),
        })

    manifest = pd.DataFrame(rows)

    if read_sizes and not manifest.empty:
        sizes = []
        for index, path in enumerate(manifest["image_path"], start=1):
            if verbose and index % 1000 == 0:
                print(f"    read {index}/{len(manifest)} image headers")
            sizes.append(read_image_size(Path(path)))
        manifest["width"] = [size[0] for size in sizes]
        manifest["height"] = [size[1] for size in sizes]

    manifest.attrs["skipped"] = skipped
    return manifest[data_config["manifest_columns"]]


def check_expected(manifest: pd.DataFrame) -> list[str]:
    """Compare the manifest against the counts recorded in configs/data.yaml.

    Returns a list of human-readable problems. Empty list means the dataset is exactly
    what every downstream result was computed against.
    """
    expected = load_config("data")["expected"]
    tolerance = expected.get("tolerance", 0)
    problems: list[str] = []

    def compare(what: str, actual: int, target: int) -> None:
        if abs(actual - target) > tolerance:
            problems.append(f"{what}: got {actual}, expected {target}")

    compare("total images", len(manifest), expected["images"])
    compare("total patients", manifest["patient_id"].nunique(), expected["patients"])

    per_class_images = manifest["label"].value_counts()
    per_class_patients = manifest.groupby("label")["patient_id"].nunique()
    for code, target in expected["images_per_class"].items():
        compare(f"images[{code}]", int(per_class_images.get(code, 0)), target)
    for code, target in expected["patients_per_class"].items():
        compare(f"patients[{code}]", int(per_class_patients.get(code, 0)), target)

    return problems


def summarise(manifest: pd.DataFrame) -> dict:
    """Descriptive statistics for the report and for the audit trail."""
    by_class = manifest.groupby("label").agg(
        images=("image_id", "size"),
        patients=("patient_id", "nunique"),
        multi_finding=("multi_finding", "sum"),
        mean_age=("age", "mean"),
    )
    by_class["multi_finding_pct"] = (
        by_class["multi_finding"] / by_class["images"] * 100
    ).round(1)
    by_class["mean_age"] = by_class["mean_age"].round(1)

    images_per_patient = manifest.groupby("patient_id").size()
    patient_classes = manifest.groupby("patient_id")["label"].nunique()

    return {
        "total_images": int(len(manifest)),
        "total_patients": int(manifest["patient_id"].nunique()),
        "by_class": by_class,
        "eye_counts": manifest["eye"].value_counts().to_dict(),
        "patients_with_both_eyes": int((images_per_patient == 2).sum()),
        "patients_with_one_eye": int((images_per_patient == 1).sum()),
        "patients_spanning_two_classes": int((patient_classes > 1).sum()),
        "missing_age": int(manifest["age"].isna().sum()),
        "missing_sex": int(manifest["sex"].isna().sum()),
        "unreadable_images": int(manifest["width"].isna().sum()),
        "distinct_resolutions": int(
            manifest.dropna(subset=["width"])
            .apply(lambda r: f"{int(r.width)}x{int(r.height)}", axis=1)
            .nunique()
        ) if manifest["width"].notna().any() else 0,
    }


def write_manifest(manifest: pd.DataFrame) -> Path:
    path = output_dir("manifests") / "odir_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8")
    return path
