"""Synthetic ophthalmic referral corpus with machine-readable ground truth.

WHAT THIS IS AND IS NOT
-----------------------
These are generated documents. They support a technical claim about OCR and field
extraction under controlled degradation. They are NOT clinical data and nothing measured
on them is clinical validation. Every number derived from this corpus carries that caveat.

Demographics (age, sex, laterality) are taken from real ODIR patient rows, so half the
document content is real metadata; the clinical narrative is generated.

TWO DESIGN RULES THAT MAKE THE EVALUATION MEAN ANYTHING
-------------------------------------------------------
1. SEALED LAYOUTS. Extractors are developed against layouts A-C only. D-F are rendered to
   a separate directory and opened once, after the extractors are frozen. Without this the
   generator and the extractor share a template and extraction F1 measures nothing.

2. DISTRACTORS AND ABSENCES. Some fields are genuinely missing, and some non-target fields
   are shaped like targets (an insurance id that looks like an IOP reading, an address
   containing a date). Precision has to be earned rather than handed over.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- vocabulary -----------------------------------------------------------------------
DIAGNOSIS_TEXT = {
    "N": ["No abnormality detected", "Normal fundus", "Within normal limits"],
    "D": ["Diabetic retinopathy", "Non-proliferative diabetic retinopathy",
          "NPDR, moderate"],
    "G": ["Glaucoma", "Primary open angle glaucoma", "Glaucoma suspect"],
    "C": ["Cataract", "Immature senile cataract", "Nuclear sclerotic cataract"],
}
# Synonyms per field label, so extraction has to generalise over phrasing, not just pixels.
LABELS = {
    "patient_id": ["Patient ID", "MRN", "Record No", "Patient No"],
    "age": ["Age", "Age (yrs)", "Patient Age"],
    "sex": ["Sex", "Gender"],
    "eye": ["Eye", "Laterality", "Side"],
    "exam_date": ["Date", "Exam Date", "Date of Examination"],
    "visual_acuity": ["Visual Acuity", "VA", "BCVA", "Vision"],
    "iop": ["IOP", "Intraocular Pressure", "Tension"],
    "stated_diagnosis": ["Diagnosis", "Impression", "Clinical Impression"],
    "referral_reason": ["Reason for Referral", "Referral Reason", "Indication"],
}
EYE_TEXT = {"left": ["Left", "OS", "L"], "right": ["Right", "OD", "R"]}
SEX_TEXT = {"Male": ["Male", "M"], "Female": ["Female", "F"]}
REFERRAL_REASONS = ["Routine diabetic screening", "Blurred vision", "Reduced acuity",
                    "Raised intraocular pressure", "Annual review", "Glare and haloes"]
VA_VALUES = ["6/6", "6/9", "6/12", "6/18", "6/24", "6/36", "6/60"]
CLINICS = ["City Eye Hospital", "Sunrise Vision Centre", "Community Eye Care Unit",
           "Regional Ophthalmology Dept"]
DOCTORS = ["Dr A Sharma", "Dr R Menon", "Dr S Gupta", "Dr K Rao", "Dr P Banerjee"]


@dataclass
class Document:
    document_id: str
    layout: str
    degradation_level: int
    patient_id: str
    true_label: str                    # the ODIR label for the paired image
    stated_diagnosis_label: str        # what the document CLAIMS - may be corrupted
    diagnosis_corrupted: bool
    fields: dict                       # ground truth, None where genuinely absent
    distractors: list = field(default_factory=list)
    image_path: str = ""


def _font(size: int, bold: bool = False):
    """DejaVu ships with matplotlib, so no system font install is needed."""
    try:
        import matplotlib
        base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(base / name), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def sample_record(rng: random.Random, patient_row: dict, label: str,
                  corrupt_probability: float, classes: list[str],
                  optional_fields: list[str], missing_rate: float) -> Document:
    """One document's ground truth, built around a real ODIR patient row."""
    stated_label = label
    corrupted = False
    if rng.random() < corrupt_probability:
        alternatives = [c for c in classes if c != label]
        stated_label = rng.choice(alternatives)
        corrupted = True

    fields = {
        "patient_id": str(patient_row["patient_id"]),
        "age": str(int(patient_row["age"])),
        "sex": rng.choice(SEX_TEXT[patient_row["sex"]]),
        "eye": rng.choice(EYE_TEXT[patient_row["eye"]]),
        "exam_date": f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026",
        "visual_acuity": rng.choice(VA_VALUES),
        "iop": f"{rng.randint(10, 26)}",
        "stated_diagnosis": rng.choice(DIAGNOSIS_TEXT[stated_label]),
        "referral_reason": rng.choice(REFERRAL_REASONS),
    }
    # Genuine absences, so recall and precision both mean something.
    for name in optional_fields:
        if rng.random() < missing_rate:
            fields[name] = None

    distractors = []
    if rng.random() < 0.6:
        distractors.append(("Insurance ID", f"{rng.randint(10, 26)}-{rng.randint(1000, 9999)}"))
    if rng.random() < 0.5:
        distractors.append(("Clinic Address",
                            f"{rng.randint(1, 99)} Station Road, est. {rng.randint(1, 28):02d}/"
                            f"{rng.randint(1, 12):02d}/1998"))
    if rng.random() < 0.4:
        distractors.append(("Appointment Ref", f"{rng.choice(['6/', '20/'])}{rng.randint(10, 99)}"))

    return Document(
        document_id="", layout="", degradation_level=0,
        patient_id=str(patient_row["patient_id"]), true_label=label,
        stated_diagnosis_label=stated_label, diagnosis_corrupted=corrupted,
        fields=fields, distractors=distractors,
    )


# --- layouts --------------------------------------------------------------------------
# Each returns a rendered RGB image. The visual variety is the point: an extractor tuned to
# "Label: value on its own line" must cope with columns, tables and prose.

def _blank(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (width, height), "white")
    return canvas, ImageDraw.Draw(canvas)


def _present(document: Document, rng: random.Random) -> list[tuple[str, str]]:
    return [(rng.choice(LABELS[k]), v) for k, v in document.fields.items() if v is not None]


def layout_key_value(document: Document, rng: random.Random) -> Image.Image:
    items = _present(document, rng) + document.distractors
    canvas, draw = _blank(1000, 150 + 46 * len(items))
    draw.text((40, 34), rng.choice(CLINICS).upper(), fill="black", font=_font(30, True))
    draw.line((40, 84, 960, 84), fill="black", width=2)
    for i, (label, value) in enumerate(items):
        draw.text((40, 108 + i * 46), f"{label}: {value}", fill="black", font=_font(24))
    return canvas


def layout_two_column(document: Document, rng: random.Random) -> Image.Image:
    items = _present(document, rng) + document.distractors
    rows = (len(items) + 1) // 2
    canvas, draw = _blank(1100, 160 + 48 * rows)
    draw.text((40, 34), rng.choice(CLINICS), fill="black", font=_font(28, True))
    draw.line((40, 82, 1060, 82), fill="black", width=2)
    for i, (label, value) in enumerate(items):
        x = 40 if i % 2 == 0 else 580
        y = 110 + (i // 2) * 48
        draw.text((x, y), f"{label}:", fill="black", font=_font(21, True))
        draw.text((x + 210, y), str(value), fill="black", font=_font(21))
    return canvas


def layout_letterhead(document: Document, rng: random.Random) -> Image.Image:
    items = _present(document, rng) + document.distractors
    canvas, draw = _blank(1000, 250 + 44 * len(items))
    draw.text((40, 30), rng.choice(CLINICS), fill="black", font=_font(32, True))
    draw.text((40, 74), "Ophthalmology Referral", fill="black", font=_font(22))
    draw.text((40, 106), f"Referring clinician: {rng.choice(DOCTORS)}", fill="black", font=_font(19))
    draw.line((40, 140, 960, 140), fill="black", width=3)
    draw.text((40, 158), "Dear Colleague, please see the following patient.",
              fill="black", font=_font(19))
    for i, (label, value) in enumerate(items):
        draw.text((70, 200 + i * 44), f"{label} - {value}", fill="black", font=_font(22))
    return canvas


def layout_table(document: Document, rng: random.Random) -> Image.Image:
    items = _present(document, rng) + document.distractors
    canvas, draw = _blank(1000, 170 + 44 * len(items))
    draw.text((40, 30), "OPHTHALMIC ASSESSMENT RECORD", fill="black", font=_font(26, True))
    top = 90
    draw.rectangle((40, top, 960, top + 40 + 44 * len(items)), outline="black", width=2)
    draw.line((400, top, 400, top + 40 + 44 * len(items)), fill="black", width=2)
    draw.line((40, top + 40, 960, top + 40), fill="black", width=2)
    draw.text((52, top + 8), "FIELD", fill="black", font=_font(20, True))
    draw.text((412, top + 8), "VALUE", fill="black", font=_font(20, True))
    for i, (label, value) in enumerate(items):
        y = top + 48 + i * 44
        draw.text((52, y), str(label), fill="black", font=_font(20))
        draw.text((412, y), str(value), fill="black", font=_font(20))
        draw.line((40, y + 38, 960, y + 38), fill="#999999", width=1)
    return canvas


def layout_narrative(document: Document, rng: random.Random) -> Image.Image:
    f = document.fields
    parts = [f"This {f['age']}-year-old {f['sex'].lower()} patient "
             f"(record {f['patient_id']}) attended on {f['exam_date']}."]
    parts.append(f"The {f['eye']} eye was examined.")
    if f.get("visual_acuity"):
        parts.append(f"Visual acuity was recorded as {f['visual_acuity']}.")
    if f.get("iop"):
        parts.append(f"Intraocular pressure measured {f['iop']} mmHg.")
    parts.append(f"Impression: {f['stated_diagnosis']}.")
    if f.get("referral_reason"):
        parts.append(f"Referral was made for {f['referral_reason'].lower()}.")
    for label, value in document.distractors:
        parts.append(f"{label} {value}.")

    # Wrap by hand so the field values land mid-sentence, which is the hard case.
    lines, current = [], ""
    for word in " ".join(parts).split():
        if len(current) + len(word) > 74:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)

    canvas, draw = _blank(1000, 190 + 40 * len(lines))
    draw.text((40, 34), rng.choice(CLINICS), fill="black", font=_font(28, True))
    draw.text((40, 78), "Clinical Note", fill="black", font=_font(21))
    draw.line((40, 112, 960, 112), fill="black", width=2)
    for i, line in enumerate(lines):
        draw.text((40, 140 + i * 40), line, fill="black", font=_font(22))
    return canvas


def layout_margin_notes(document: Document, rng: random.Random) -> Image.Image:
    items = _present(document, rng)
    canvas, draw = _blank(1150, 170 + 46 * len(items))
    draw.text((40, 30), rng.choice(CLINICS), fill="black", font=_font(27, True))
    draw.line((40, 78, 820, 78), fill="black", width=2)
    draw.line((840, 20, 840, 150 + 46 * len(items)), fill="#666666", width=1)
    draw.text((860, 34), "NOTES", fill="#444444", font=_font(18, True))
    for i, (label, value) in enumerate(items):
        draw.text((40, 104 + i * 46), f"{label}: {value}", fill="black", font=_font(23))
    for i, (label, value) in enumerate(document.distractors):
        draw.text((860, 70 + i * 40), f"{label} {value}", fill="#444444", font=_font(16))
    return canvas


LAYOUT_RENDERERS = {
    "A": layout_key_value, "B": layout_two_column, "C": layout_letterhead,
    "D": layout_table, "E": layout_narrative, "F": layout_margin_notes,
}


def degrade(image: Image.Image, level: int, params: dict, rng: random.Random) -> Image.Image:
    """Apply the exact parameters recorded in configs/idp.yaml for this level."""
    if level == 0:
        return image
    settings = params[level]
    if settings["rotation_deg"]:
        image = image.rotate(rng.uniform(-settings["rotation_deg"], settings["rotation_deg"]),
                             resample=Image.BICUBIC, fillcolor="white")
    if settings["downscale"] < 1.0:
        small = (max(1, int(image.width * settings["downscale"])),
                 max(1, int(image.height * settings["downscale"])))
        image = image.resize(small, Image.LANCZOS).resize(image.size, Image.LANCZOS)
    if settings["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(radius=settings["blur"]))
    if settings["noise_sigma"]:
        generator = np.random.default_rng(rng.randint(0, 10 ** 6))
        noise = generator.normal(0, settings["noise_sigma"], (image.height, image.width, 3))
        image = Image.fromarray(
            np.clip(np.array(image).astype(float) + noise, 0, 255).astype(np.uint8))
    return image


def render(document: Document, layout: str, level: int, params: dict,
           out_dir: Path, rng: random.Random) -> Document:
    image = LAYOUT_RENDERERS[layout](document, rng)
    image = degrade(image, level, params, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{document.document_id}.jpg"
    image.save(path, quality=params[level]["jpeg_quality"] if level else 95)
    document.layout = layout
    document.degradation_level = level
    document.image_path = str(path)
    return document


def write_ground_truth(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(d) for d in documents], indent=2, default=str), encoding="utf-8")
