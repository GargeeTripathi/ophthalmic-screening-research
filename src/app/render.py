"""HTML rendering for the screening interface.

Implements the Figma design (file u8Mu72qiltp4zVrJWTKHWP, page "Screening UI") in the
browser. The tokens below are the same ones defined as Figma variables, so the design and
the running system stay one system rather than two that resemble each other.

THE RULE THE LAYOUT ENCODES
---------------------------
The image pathway and the document pathway are separate columns that meet only at the
decision banner, because the document never changes the finding — it only affects whether
the system answers. And the deferral is stated before any class name: a case the system
declined to answer must never read as an answer.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Mirrors the Figma variable collection "Screening tokens".
CSS = """
:root{
  --ground:#F1F4F3; --surface:#fff; --surface-2:#E8EDEB;
  --ink:#141D1C; --ink-2:#3B4B48; --muted:#64756F; --rule:#D2DAD7;
  --accent:#0E5C58; --accent-soft:#D8E7E4;
  --defer:#A32118; --defer-bg:#F7E2E0;
  --warn:#B85A11; --warn-bg:#F9EADB;
  --accept:#3F6B3A; --accept-bg:#E4EFE2;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:14px/1.45 Inter,-apple-system,Segoe UI,system-ui,sans-serif}
a{color:var(--accent)}
header{background:var(--surface);border-bottom:1px solid var(--rule);padding:24px 32px}
.eyebrow{font-size:10px;font-weight:600;letter-spacing:.12em;color:var(--muted);
  text-transform:uppercase;margin:0}
h1{font-size:26px;font-weight:600;letter-spacing:-.02em;margin:4px 0 0}
.sub{color:var(--ink-2);font-size:13px;margin:4px 0 0;max-width:80ch}
.wrap{padding:20px 32px 32px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;
  margin-bottom:20px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 18px}
.tile b{display:block;font-size:26px;font-weight:600;line-height:1.15}
.tile span{font-size:11px;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--surface);
  border:1px solid var(--rule);border-radius:10px;overflow:hidden}
th{background:var(--surface-2);color:var(--muted);font-size:10px;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;text-align:left;padding:12px 20px}
td{padding:14px 20px;border-bottom:1px solid var(--rule);font-size:12px;color:var(--ink-2)}
tr.defer td{background:var(--defer-bg)}
tr:last-child td{border-bottom:0}
td.case{font-weight:500;color:var(--ink)}
td.reason{font-size:11px}
tr.defer td.reason{color:var(--defer)}
.chip{display:inline-block;font-size:10px;font-weight:600;letter-spacing:.08em;
  padding:5px 9px;border-radius:4px}
.chip.defer{background:var(--defer);color:#fff}
.chip.accept{background:var(--accept-bg);color:var(--accept)}
.chip.warn{background:var(--warn-bg);color:var(--warn)}
.banner{display:flex;gap:16px;align-items:center;background:var(--defer-bg);
  border-left:5px solid var(--defer);border-radius:8px;padding:16px 20px;margin-bottom:20px}
.banner.accept{background:var(--accept-bg);border-left-color:var(--accept)}
.banner b{display:block;font-size:14px;color:var(--defer)}
.banner.accept b{color:var(--accept)}
.banner p{margin:3px 0 0;font-size:12px;color:var(--ink-2)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}
@media(max-width:1000px){.cols{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:18px 20px;margin-bottom:20px}
.label{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin:0 0 12px}
.finding{font-size:20px;font-weight:600;letter-spacing:-.01em;margin:0}
.conf{font-size:22px;font-weight:600;color:var(--accent);margin:8px 0 0}
.note{font-size:11px;color:var(--muted);margin:2px 0 0}
img.fundus{width:100%;border-radius:8px;display:block;background:#16100c}
img.doc{width:100%;border-radius:6px;border:1px solid var(--rule);display:block}
.bar{display:grid;grid-template-columns:150px 1fr 44px;gap:12px;align-items:center;
  margin:8px 0;font-size:12px}
.track{height:10px;border-radius:99px;background:var(--surface-2);overflow:hidden}
.track i{display:block;height:10px;border-radius:99px;background:var(--accent)}
.track i.dim{background:var(--muted)}
.fields{width:100%;border-collapse:collapse;border:0;background:none}
.fields td{padding:8px 0;border-bottom:1px solid var(--rule);font-size:12px}
.fields td:first-child{color:var(--muted);width:150px}
.rel{display:flex;gap:10px;align-items:center;background:var(--warn-bg);
  border-radius:6px;padding:10px 12px;margin-top:12px}
.rel b{font-size:15px;color:var(--warn)}
.rel span{font-size:11px;color:var(--ink-2)}
.disc{background:var(--defer-bg);border-radius:6px;padding:12px 14px}
.disc.ok{background:var(--accept-bg)}
.disc b{font-size:11px;font-weight:600;letter-spacing:.08em;color:var(--defer)}
.disc.ok b{color:var(--accept)}
.disc p{margin:6px 0 0;font-size:12px;color:var(--ink-2)}
.qa{border-top:1px solid var(--rule);padding:10px 0}
.qa b{display:block;font-size:12px}
.qa p{margin:4px 0 0;font-size:12px;color:var(--ink-2)}
footer{background:var(--defer-bg);border-top:1px solid var(--rule);padding:18px 32px 24px}
footer b{font-size:12px;color:var(--defer)}
footer p{margin:6px 0 0;font-size:11px;color:var(--ink-2);max-width:110ch}
form.upload{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;
  background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:16px 20px;margin-bottom:20px}
form.upload label{display:block;font-size:10px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin-bottom:6px}
form.upload input[type=file]{font-size:12px}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;
  padding:10px 18px;font-size:13px;font-weight:500;cursor:pointer}
button:hover{filter:brightness(1.1)}
.back{font-size:12px;text-decoration:none}
"""

DISCLAIMER_TITLE = "Academic screening prototype. Not a medical device and not a diagnosis."
DISCLAIMER_BODY = (
    "A qualified clinician reviews every case and makes every decision. Trained on a public "
    "research dataset with 203 cataract and 182 glaucoma patients. Referral documents used "
    "in development were synthetic. No prospective clinical validation."
)


def thumbnail(path: str | Path, size: int = 520) -> str:
    """Inline an image so a page is one portable file."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001 - a missing image must not break the page
        return ""


def _page(title: str, subtitle: str, body: str, upload: bool = False) -> str:
    form = ("""
  <form class=upload method=post action="/screen" enctype="multipart/form-data">
    <div><label>Fundus photograph</label><input type=file name=image accept="image/*" required></div>
    <div><label>Referral document (optional)</label><input type=file name=document accept="image/*"></div>
    <button type=submit>Screen this case</button>
  </form>""" if upload else "")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header>
  <p class=eyebrow>Ophthalmic screening support</p>
  <h1>{html.escape(title)}</h1>
  <p class=sub>{html.escape(subtitle)}</p>
</header>
<div class=wrap>{form}{body}</div>
<footer><b>{DISCLAIMER_TITLE}</b><p>{DISCLAIMER_BODY}</p></footer>
</body></html>"""


def worklist(cases: list[dict]) -> str:
    deferred = sum(1 for c in cases if c["decision"] == "DEFER")
    discordant = sum(1 for c in cases if c.get("concordant") is False)

    rows = []
    for position, case in enumerate(cases, start=1):
        is_defer = case["decision"] == "DEFER"
        reason = "; ".join(case["deferral_reasons"]) if is_defer else "—"
        rows.append(f"""<tr class="{'defer' if is_defer else ''}">
  <td>{position}</td>
  <td class=case><a href="/case/{position - 1}">{html.escape(case['image_id'])}</a></td>
  <td>{html.escape(case['predicted_name'].replace('_', ' '))}</td>
  <td>{case['confidence']:.0%}</td>
  <td><span class="chip {'defer' if is_defer else 'accept'}">{case['decision']}</span></td>
  <td class=reason>{html.escape(reason)}</td>
</tr>""")

    body = f"""
<div class=tiles>
  <div class=tile><b>{len(cases)}</b><span>cases in queue</span></div>
  <div class=tile><b style="color:var(--defer)">{deferred}</b><span>deferred for review</span></div>
  <div class=tile><b style="color:var(--accept)">{len(cases) - deferred}</b><span>accepted</span></div>
  <div class=tile><b style="color:var(--warn)">{discordant}</b><span>image–referral discordant</span></div>
</div>
<table>
  <tr><th>#</th><th>Case</th><th>Screening finding</th><th>Conf.</th><th>Status</th><th>Reason</th></tr>
  {''.join(rows)}
</table>
<p class=note style="margin-top:14px">The referral document never changes a finding — it only
affects whether the system answers.</p>"""

    return _page("Screening worklist",
                 "Ordered by urgency: deferred cases first, then by ascending confidence. "
                 "Queue order is what the system changes — every case is still read by a "
                 "clinician.", body, upload=True)


def case_review(case: dict, chatbot_summary: list[tuple[str, str]]) -> str:
    is_defer = case["decision"] == "DEFER"
    reasons = "; ".join(case["deferral_reasons"]) if is_defer else (
        "Image confidence is above the threshold and the referral does not contradict it.")

    bars = []
    ordered = sorted(case["probabilities"].items(), key=lambda kv: -kv[1])
    for code, value in ordered:
        top = code == case["predicted_class"]
        bars.append(f"""<div class=bar>
  <span>{html.escape(case['class_display'][code])}</span>
  <span class=track><i class="{'' if top else 'dim'}" style="width:{value * 100:.0f}%"></i></span>
  <span style="text-align:right">{value:.0%}</span>
</div>""")

    if case["document_present"]:
        rows = "".join(
            f"<tr><td>{html.escape(k.replace('_', ' '))}</td>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in case["document_fields"].items())
        confidence = case.get("extraction_confidence")
        rel = ("" if confidence is None else f"""<div class=rel><b>{confidence:.0%}</b>
<span>estimated probability the diagnosis field was read correctly. Raw OCR confidence is
not used for this — it scores AUROC 0.52.</span></div>""")
        doc_image = (f"<img class=doc src='{thumbnail(case['document_path'], 480)}' "
                     "alt='referral document' style='margin-bottom:12px'>"
                     if case.get("document_path") else "")
        document_card = f"""<div class=card><p class=label>Extracted from the referral</p>
{doc_image}<table class=fields>{rows}</table>{rel}</div>"""

        if case.get("concordant") is None:
            concordance = """<div class=card><p class=label>Image vs referral</p>
<div class=disc><b>NO COMPARABLE DIAGNOSIS</b><p>The referral did not yield a diagnosis this
system can compare against, so concordance could not be assessed.</p></div></div>"""
        elif case["concordant"]:
            concordance = f"""<div class=card><p class=label>Image vs referral</p>
<div class="disc ok"><b>CONCORDANT</b><p>The referral agrees with the image finding. The
image assigns {case['agreement']:.0%} probability to that class.</p></div></div>"""
        else:
            concordance = f"""<div class=card><p class=label>Image vs referral</p>
<div class=disc><b>DISCORDANT</b><p>The referral states
{html.escape(case['class_display'].get(case['stated_diagnosis_class'], 'something else'))},
but the image finding is
{html.escape(case['class_display'][case['predicted_class']])}. The image assigns only
{case['agreement']:.0%} probability to the referral's claim. Either source may be wrong — a
clinician should review.</p></div></div>"""
    else:
        document_card = """<div class=card><p class=label>Referral document</p>
<p class=note>No referral was supplied. The screening finding is unaffected — the document
only ever influences the deferral decision.</p></div>"""
        concordance = ""

    briefing = "".join(
        f"<div class=qa><b>{html.escape(q)}?</b><p>{html.escape(a)}</p></div>"
        for q, a in chatbot_summary)

    body = f"""
<p><a class=back href="/">&larr; back to the worklist</a></p>
<div class="banner {'' if is_defer else 'accept'}">
  <span class="chip {'defer' if is_defer else 'accept'}">{case['decision']}</span>
  <div><b>{'Referred to clinician' if is_defer else 'Accepted for reporting'}</b>
  <p>{html.escape(reasons)}</p></div>
</div>
<div class=cols>
  <div>
    <div class=card><p class=label>Fundus photograph</p>
      <img class=fundus src="{thumbnail(case['image_path'])}" alt="fundus photograph">
      <p class=note>{html.escape(case['image_id'])} · field of view cropped, 224×224</p>
    </div>
    <div class=card><p class=label>Screening finding · image only</p>
      <p class=finding>{html.escape(case['class_display'][case['predicted_class']])}</p>
      <p class=conf>{case['confidence']:.0%}</p>
      <p class=note>{html.escape(case['calibration_note'])}</p>
      <p class=label style="margin-top:16px">Class probabilities</p>
      {''.join(bars)}
    </div>
  </div>
  <div>
    {document_card}
    {concordance}
    <div class=card><p class=label>Grounded assistant</p>{briefing}
      <p class=note style="margin-top:10px">Answers are assembled from this case's structured
      outputs only. The assistant cannot generate clinical text and refuses questions it has
      no field for.</p>
    </div>
  </div>
</div>"""

    return _page("Case review",
                 "The referral document never changes the finding. It only affects whether "
                 "the system answers.", body)
