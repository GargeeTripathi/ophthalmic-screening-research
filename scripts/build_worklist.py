"""Build the clinician-facing worklist demo: a single self-contained HTML page.

This is the demonstration layer. It shows what the system produces per case - screening
finding, calibrated confidence, deferral state and reason, extracted referral fields,
extraction reliability, and the grounded chatbot briefing.

It is an interface, not a research contribution, and the page says so on every screen.

    python scripts/build_worklist.py

Writes reports/worklist.html - openable in any browser, no server needed.
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.chatbot import CLASS_NAMES, DISCLAIMER, Chatbot  # noqa: E402
from src.utils.config import output_dir  # noqa: E402

MAX_CASES = 24
THUMB = 260


def thumbnail(path: str) -> str:
    """Inline the image so the page is a single portable file."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((THUMB, THUMB))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return ""


def build_cases(limit: int = MAX_CASES) -> list[dict]:
    processed, manifests, documents = (output_dir("processed"), output_dir("manifests"),
                                       output_dir("documents"))
    table = pd.read_csv(processed / "deferral_table.csv", encoding="utf-8")
    manifest = pd.read_csv(manifests / "odir_manifest.csv",
                           encoding="utf-8").set_index("image_id")

    ground_truth = {}
    for split in ("dev", "sealed"):
        path = documents / f"ground_truth_{split}.json"
        if path.is_file():
            for record in json.loads(path.read_text(encoding="utf-8")):
                ground_truth[record["document_id"]] = record

    test = table[table["split"] == "sealed"].copy()
    threshold = test["c_hat"].median() if "c_hat" in test else 0.5

    # Worklist order: deferred cases first, then by ascending confidence. That IS the
    # product - the queue order is what the system actually changes in a clinic.
    test["deferred"] = (test["max_prob"] < 0.60) | (test["agreement"] < 0.25)
    test = test.sort_values(["deferred", "max_prob"], ascending=[False, True])

    cases = []
    for row in test.head(limit).itertuples():
        record = ground_truth.get(row.document_id, {})
        reasons = []
        if row.max_prob < 0.60:
            reasons.append(f"image confidence {row.max_prob:.0%} below the 60% threshold")
        if row.agreement < 0.25:
            reasons.append("referral diagnosis contradicts the image finding")
        if getattr(row, "c_hat", 1.0) < threshold:
            reasons.append("referral fields were read with low reliability")

        case = {
            "image_id": row.image_id, "document_id": row.document_id,
            "y_pred": row.y_pred, "y_true": row.y_true,
            "max_prob": float(row.max_prob), "agreement": float(row.agreement),
            "stated_label": row.stated_label,
            "deferred": bool(row.deferred),
            "deferral_reason": "; ".join(reasons) if reasons else "",
            "extraction_confidence": float(getattr(row, "c_hat", float("nan"))),
            "document_fields": record.get("fields", {}),
            "image_path": manifest.loc[row.image_id, "image_path"]
            if row.image_id in manifest.index else "",
            "document_image": record.get("image_path", ""),
            "degradation_level": int(row.degradation_level),
        }
        cases.append(case)
    return cases


def render(cases: list[dict]) -> str:
    deferred = sum(1 for c in cases if c["deferred"])
    rows = []
    for case in cases:
        chat = Chatbot(case)
        briefing = "".join(
            f"<div class=qa><div class=q>{html.escape(q)}?</div>"
            f"<div class=a>{html.escape(a)}</div></div>"
            for q, a in chat.summary())
        fields = "".join(
            f"<tr><td>{html.escape(k.replace('_', ' '))}</td>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in (case["document_fields"] or {}).items() if v is not None)
        badge = ("<span class='chip defer'>DEFER</span>" if case["deferred"]
                 else "<span class='chip accept'>ACCEPT</span>")
        agree = ("<span class='chip warn'>DISCORDANT</span>"
                 if case["stated_label"] != case["y_pred"] else "")

        rows.append(f"""
<article class=case>
  <div class=head>
    <div>{badge} {agree}
      <h3>{html.escape(case['image_id'])}</h3>
      <p class=finding>{html.escape(CLASS_NAMES.get(case['y_pred'], case['y_pred']))}
         &middot; confidence {case['max_prob']:.0%}</p>
      {"<p class=reason>" + html.escape(case['deferral_reason']) + "</p>" if case['deferral_reason'] else ""}
    </div>
  </div>
  <div class=body>
    <div class=imgs>
      <figure><img src="{thumbnail(case['image_path'])}" alt="fundus photograph">
        <figcaption>fundus</figcaption></figure>
      <figure><img src="{thumbnail(case['document_image'])}" alt="referral document">
        <figcaption>referral (degradation L{case['degradation_level']})</figcaption></figure>
    </div>
    <div class=panel>
      <h4>Extracted referral fields</h4>
      <table>{fields or '<tr><td colspan=2>none extracted</td></tr>'}</table>
      <h4>Grounded briefing</h4>
      {briefing}
    </div>
  </div>
</article>""")

    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Screening worklist</title>
<style>
:root{{--ink:#141D1C;--muted:#64756F;--rule:#D2DAD7;--accent:#0E5C58;
--crit:#A32118;--warn:#B85A11;--bg:#F1F4F3;--surface:#fff}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:0 0 4rem}}
header{{background:var(--surface);border-bottom:1px solid var(--rule);padding:1.6rem 1.4rem}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 1.4rem}}
h1{{margin:0;font-size:1.5rem;letter-spacing:-.02em}}
.sub{{color:var(--muted);margin:.4rem 0 0;max-width:60ch}}
.stats{{display:flex;gap:2rem;margin-top:1rem;font-size:.85rem;color:var(--muted)}}
.stats b{{display:block;font-size:1.4rem;color:var(--ink)}}
.banner{{background:#F7E2E0;color:#7A1912;padding:.7rem 1rem;border-radius:6px;
margin:1.2rem 0;font-size:.85rem}}
.case{{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
margin:1rem 0;overflow:hidden}}
.head{{padding:1rem 1.2rem;border-bottom:1px solid var(--rule)}}
.head h3{{margin:.4rem 0 .1rem;font-size:1.05rem;font-family:ui-monospace,monospace}}
.finding{{margin:0;color:var(--muted);font-size:.9rem}}
.reason{{margin:.4rem 0 0;color:var(--crit);font-size:.83rem}}
.body{{display:grid;gap:1.2rem;padding:1.2rem;grid-template-columns:1fr}}
@media(min-width:820px){{.body{{grid-template-columns:290px 1fr}}}}
.imgs{{display:flex;gap:.6rem;flex-wrap:wrap}}
figure{{margin:0}} img{{max-width:130px;border-radius:6px;border:1px solid var(--rule)}}
figcaption{{font-size:.72rem;color:var(--muted);margin-top:.2rem}}
.chip{{display:inline-block;font:600 .68rem/1 ui-monospace,monospace;letter-spacing:.08em;
padding:.35em .6em;border-radius:4px}}
.accept{{background:#E4EFE2;color:#2E5A2A}} .defer{{background:#F7E2E0;color:var(--crit)}}
.warn{{background:#F9EADB;color:var(--warn)}}
h4{{margin:1rem 0 .4rem;font-size:.82rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td{{padding:.25rem .5rem;border-bottom:1px solid var(--rule)}}
td:first-child{{color:var(--muted);width:11rem}}
.qa{{border-top:1px solid var(--rule);padding:.5rem 0}}
.q{{font-weight:600;font-size:.83rem}} .a{{color:#3B4B48;font-size:.85rem;margin-top:.15rem}}
footer{{color:var(--muted);font-size:.78rem;margin-top:2rem;padding-top:1rem;
border-top:1px solid var(--rule)}}
</style></head><body>
<header><div class=wrap>
<h1>Screening worklist</h1>
<p class=sub>Ordered by urgency: deferred cases first, then by ascending confidence.
The referral document never changes the screening finding &mdash; it only affects whether
the system answers at all.</p>
<div class=stats>
  <div><b>{len(cases)}</b>cases shown</div>
  <div><b>{deferred}</b>deferred for review</div>
  <div><b>{len(cases) - deferred}</b>accepted</div>
</div></div></header>
<div class=wrap>
<div class=banner><strong>{DISCLAIMER}</strong></div>
{''.join(rows)}
<footer>Generated by scripts/build_worklist.py from data/processed/deferral_table.csv.
Referral documents are synthetic. Not a medical device.</footer>
</div></body></html>"""


def main() -> int:
    print("=" * 72)
    print("BUILD WORKLIST DEMO")
    print("=" * 72)
    cases = build_cases()
    if not cases:
        print("No cases available - run scripts/run_deferral.py first.")
        return 2
    path = output_dir("reports") / "worklist.html"
    path.write_text(render(cases), encoding="utf-8")
    deferred = sum(1 for c in cases if c["deferred"])
    print(f"\n{len(cases)} cases  ({deferred} deferred, {len(cases) - deferred} accepted)")
    print(f"WRITTEN {path}  ({path.stat().st_size / 1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
