"""Acquisition-provenance features for the merged four-class fundus benchmark.

This module answers one question: given only how a file was produced and encoded, which
acquisition source did it come from?

THE CRITICAL RULE
-----------------
Source is assigned from acquisition-observable properties ONLY - file extension, encoding,
pixel dimensions, filename convention, byte size. The class folder an image sits in is
never consulted. If source were derived from the label, the experiment "does source predict
label?" would be circular and its answer meaningless.

Everything here is deterministic and reproducible from the files on disk. No model, no
training, no randomness.

The source rules below were derived by inspecting the dataset's own structure, and they
are stated as rules rather than learned so a reader can verify every assignment by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- filename conventions -------------------------------------------------------------
# Different upstream datasets name their files differently, and the convention survives
# copying, so it is one of the strongest provenance cues available.
CONVENTIONS = {
    # "1234_left.jpg" - ODIR-5K and EyePACS both use this; extension separates them.
    "patient_eye": re.compile(r"^\d+_(left|right)\.(jpg|jpeg|png)$", re.IGNORECASE),
    # "_26_2782553.jpg" - leading underscore, index, random-looking id.
    "underscore_index_hash": re.compile(r"^_\d+_\d+\.(jpg|png)$", re.IGNORECASE),
}


def filename_convention(filename: str) -> str:
    for name, pattern in CONVENTIONS.items():
        if pattern.match(filename):
            return name
    return "other"


def resolution_bucket(width: int | None, height: int | None) -> str:
    """Coarse size class. Acquisition devices cluster by sensor resolution."""
    if width is None or height is None:
        return "unknown"
    longest = max(width, height)
    if longest <= 300:
        return "tiny_256"
    if longest <= 600:
        return "small_512"
    if longest <= 1500:
        return "medium"
    return "large"


# --- source assignment ----------------------------------------------------------------
# Ordered rules. The first match wins, so the more specific rules come first.
# Each rule is (source_name, predicate over the feature dict).
def assign_source(features: dict) -> str:
    extension = features["file_extension"]
    convention = features["filename_convention"]
    bucket = features["resolution_bucket"]

    # EyePACS diabetic-retinopathy exports are the only .jpeg files in this collection.
    if extension == ".jpeg":
        return "eyepacs_jpeg"

    # High-resolution PNGs: a separate upstream export, distinguishable by encoding alone.
    if extension == ".png":
        return "png_highres"

    # ODIR-5K derived: patient_eye naming at the 512 export size.
    if extension == ".jpg" and convention == "patient_eye" and bucket == "small_512":
        return "odir_jpg512"

    # A 256-pixel crop export with its own naming scheme.
    if extension == ".jpg" and bucket == "tiny_256":
        return "crop_jpg256"

    # Native camera resolutions kept as JPEG.
    if extension == ".jpg" and bucket in ("medium", "large"):
        return "native_jpg"

    return "unassigned"


def describe_image(path: Path) -> dict:
    """Every provenance feature for one file. Header-only read - pixels are not decoded."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode
    except Exception:  # noqa: BLE001 - a bad file becomes a null row, not a crash
        width = height = None
        image_format = mode = None

    features = {
        "original_filename": path.name,
        "file_extension": path.suffix.lower(),
        "image_format": image_format,
        "colour_mode": mode,
        "width": width,
        "height": height,
        "size_bytes": path.stat().st_size,
        "aspect_ratio": round(width / height, 4) if width and height else None,
        "filename_convention": filename_convention(path.name),
        "resolution_bucket": resolution_bucket(width, height),
    }
    features["source"] = assign_source(features)
    return features


# Feature columns handed to the E0a metadata-only probe. Deliberately excludes anything
# derived from the label, and excludes `source` itself - the probe must work from raw
# acquisition properties, not from our own rule output.
PROBE_FEATURES = [
    "file_extension",
    "image_format",
    "colour_mode",
    "width",
    "height",
    "size_bytes",
    "aspect_ratio",
    "filename_convention",
    "resolution_bucket",
]
