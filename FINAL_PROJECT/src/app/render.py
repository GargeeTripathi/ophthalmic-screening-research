"""HTML rendering for the screening interface.

Implements the Figma design (file u8Mu72qiltp4zVrJWTKHWP, page "Screening UI") in the
browser. The colour tokens below are the same ones defined as Figma variables, so the design
and the running system stay one system rather than two that resemble each other.

The dashboard chrome added since — the navigation rail, the KPI tiles and the case intake
panel — exists here but NOT yet in that Figma file, so the two have drifted by exactly those
three components. Re-sync them before treating the Figma file as the source of truth again.

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
  --rail:#0B2E2C; --rail-ink:#CFE0DD; --rail-rule:#1C4744;
  --shadow:0 1px 2px rgba(11,46,44,.06),0 8px 24px -12px rgba(11,46,44,.18);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:14px/1.45 Inter,-apple-system,Segoe UI,system-ui,sans-serif}
a{color:var(--accent)}

/* --- dashboard frame: fixed rail, scrolling main column ------------------- */
.app{display:grid;grid-template-columns:236px 1fr;min-height:100vh}
.rail{background:var(--rail);color:var(--rail-ink);display:flex;flex-direction:column;
  gap:22px;padding:22px 16px}
.brand{display:flex;gap:11px;align-items:center;padding:0 6px}
.brand .mark{width:32px;height:32px;border-radius:9px;background:var(--accent);
  display:grid;place-items:center;color:#fff;font-size:15px;flex:none}
.brand b{display:block;color:#fff;font-size:14px;font-weight:600;letter-spacing:-.01em}
.brand span{font-size:10px;letter-spacing:.06em;text-transform:uppercase;opacity:.7}
.rail nav{display:flex;flex-direction:column;gap:2px}
.rail nav a{display:flex;gap:9px;align-items:center;text-decoration:none;color:var(--rail-ink);
  font-size:13px;padding:9px 11px;border-radius:7px}
.rail nav a:hover{background:var(--rail-rule);color:#fff}
.rail nav a.on{background:var(--accent);color:#fff;font-weight:500}
.railhead{font-size:9.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  opacity:.55;padding:0 11px;margin:6px 0 2px}
.railfoot{margin-top:auto;border-top:1px solid var(--rail-rule);padding:14px 11px 0;
  font-size:11px;line-height:1.5;opacity:.72}
.railfoot b{display:block;color:#fff;font-weight:500;opacity:1}
main{min-width:0;display:flex;flex-direction:column}
main footer{margin-top:auto}          /* short pages: footer at the bottom, not mid-screen */
@media(max-width:860px){.app{grid-template-columns:1fr}
  .rail{flex-direction:row;align-items:center;flex-wrap:wrap;gap:12px;padding:14px 18px}
  .rail nav{flex-direction:row;flex-wrap:wrap}.railhead,.railfoot{display:none}}

header{background:var(--surface);border-bottom:1px solid var(--rule);padding:22px 32px;
  display:flex;gap:24px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
.eyebrow{font-size:10px;font-weight:600;letter-spacing:.12em;color:var(--muted);
  text-transform:uppercase;margin:0}
h1{font-size:24px;font-weight:600;letter-spacing:-.02em;margin:4px 0 0}
.sub{color:var(--ink-2);font-size:13px;margin:5px 0 0;max-width:80ch}
.meta{display:flex;gap:8px;flex-wrap:wrap}
.pill{display:inline-flex;gap:6px;align-items:center;background:var(--surface-2);
  border-radius:99px;padding:6px 12px;font-size:11px;color:var(--ink-2);white-space:nowrap}
.pill i{width:6px;height:6px;border-radius:99px;background:var(--accent);font-style:normal}
.wrap{padding:22px 32px 36px}

/* --- KPI tiles ------------------------------------------------------------ */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;
  margin-bottom:22px}
.tile{position:relative;background:var(--surface);border:1px solid var(--rule);
  border-radius:12px;padding:15px 18px 16px;box-shadow:var(--shadow);overflow:hidden}
.tile:before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--accent)}
.tile.is-defer:before{background:var(--defer)}
.tile.is-accept:before{background:var(--accept)}
.tile.is-warn:before{background:var(--warn)}
.tile b{display:block;font-size:28px;font-weight:600;line-height:1.1;letter-spacing:-.02em}
.tile span{display:block;font-size:11px;color:var(--muted);margin-top:3px}
.tile em{font-style:normal;font-size:10.5px;color:var(--muted);opacity:.8}

/* --- queue table ---------------------------------------------------------- */
.tablecard{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
  box-shadow:var(--shadow);overflow:hidden}
.tablecard .caption{display:flex;justify-content:space-between;align-items:center;gap:14px;
  padding:14px 20px;border-bottom:1px solid var(--rule);flex-wrap:wrap}
.tablecard .caption b{font-size:13px;font-weight:600}
.tablecard .caption span{font-size:11px;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--surface)}
th{background:var(--surface-2);color:var(--muted);font-size:10px;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;text-align:left;padding:11px 20px}
td{padding:13px 20px;border-bottom:1px solid var(--rule);font-size:12px;color:var(--ink-2);
  vertical-align:middle}
tbody tr:hover td{background:var(--surface-2)}
tr.defer td{background:var(--defer-bg)}
tr.defer:hover td{background:#F2D8D5}
tr:last-child td{border-bottom:0}
td.case{font-weight:500}
td.case a{color:var(--accent);font-weight:600;text-decoration:none}
td.case a:hover{text-decoration:underline}
td.reason{font-size:11px}
tr.defer td.reason{color:var(--defer)}
td.conf{white-space:nowrap;font-variant-numeric:tabular-nums}
/* The track needs its own outline: on a pink DEFER row a plain grey bar disappears and the
   fill reads as a stray dash rather than as a gauge. */
td.conf i{display:inline-block;width:64px;height:6px;border-radius:99px;background:#fff;
  border:1px solid var(--rule);margin-left:10px;vertical-align:middle;overflow:hidden}
td.conf i b{display:block;height:6px;border-radius:99px;background:var(--accent)}
tr.defer td.conf i b{background:var(--defer)}
.chip{display:inline-block;font-size:10px;font-weight:600;letter-spacing:.08em;
  padding:5px 9px;border-radius:4px}
.chip.defer{background:var(--defer);color:#fff}
.chip.accept{background:var(--accept-bg);color:var(--accept)}
.chip.warn{background:var(--warn-bg);color:var(--warn)}
.banner{display:flex;gap:16px;align-items:center;background:var(--defer-bg);
  border-left:5px solid var(--defer);border-radius:10px;padding:16px 20px;margin-bottom:20px;
  box-shadow:var(--shadow)}
.banner.accept{background:var(--accept-bg);border-left-color:var(--accept)}
.banner b{display:block;font-size:14px;color:var(--defer)}
.banner.accept b{color:var(--accept)}
.banner p{margin:3px 0 0;font-size:12px;color:var(--ink-2)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}
/* The document column runs much longer than the image column. Sticking the image and the
   finding keeps them beside whatever part of the referral is being read. */
.cols>div:first-child{position:sticky;top:20px}
@media(max-width:1000px){.cols{grid-template-columns:1fr}
  .cols>div:first-child{position:static}}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
  padding:18px 20px;margin-bottom:20px;box-shadow:var(--shadow)}
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
/* --- the highlighted intake panel ----------------------------------------- */
form.upload{background:linear-gradient(180deg,var(--accent-soft),var(--surface) 62%);
  border:1px solid var(--accent);border-radius:14px;padding:18px 20px 16px;
  margin-bottom:22px;box-shadow:var(--shadow)}
form.upload .uphead{display:flex;gap:12px;align-items:flex-start;margin-bottom:14px}
form.upload .upmark{width:34px;height:34px;border-radius:10px;background:var(--accent);
  color:#fff;display:grid;place-items:center;font-size:19px;line-height:1;flex:none}
form.upload .uphead b{display:block;font-size:15px;font-weight:600;letter-spacing:-.01em}
form.upload .uphead p{margin:2px 0 0;font-size:12px;color:var(--ink-2);max-width:78ch}
.drops{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.drops{grid-template-columns:1fr}}
label.drop{display:block;position:relative;background:var(--surface);
  border:1.5px dashed var(--rule);border-radius:10px;padding:14px 16px;cursor:pointer}
label.drop:hover,label.drop:focus-within{border-color:var(--accent);background:#fff}
label.drop.req{border-color:var(--accent);border-style:solid}
label.drop b{display:block;font-size:12.5px;font-weight:600}
label.drop .hint{display:block;font-size:10px;font-weight:600;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);margin-top:3px}
label.drop input[type=file]{display:block;margin-top:9px;font-size:11.5px;width:100%;
  color:var(--ink-2)}
label.drop input[type=file]::file-selector-button{background:var(--surface-2);color:var(--ink);
  border:1px solid var(--rule);border-radius:6px;padding:6px 11px;font:inherit;font-size:11.5px;
  margin-right:9px;cursor:pointer}
label.drop input[type=file]::file-selector-button:hover{background:var(--accent-soft);
  border-color:var(--accent)}
.upfoot{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px}
.upfoot .fine{font-size:11px;color:var(--muted);max-width:62ch}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;
  padding:11px 20px;font-size:13px;font-weight:500;cursor:pointer}
button:hover{filter:brightness(1.12)}
button:focus-visible,label.drop input:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}
.back{font-size:12px;text-decoration:none;display:inline-block;margin-bottom:14px}
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


UPLOAD_PANEL = """
  <form class=upload id=upload method=post action="/screen" enctype="multipart/form-data">
    <div class=uphead>
      <span class=upmark>+</span>
      <div><b>Screen a new case</b>
      <p>Choose a fundus photograph, and its referral letter if you have one. The case runs
      through the same pipeline the research evaluated, and its result is added to the top of
      the queue.</p></div>
    </div>
    <div class=drops>
      <label class="drop req"><b>Fundus photograph</b>
        <span class=hint>Required &middot; JPG or PNG</span>
        <input type=file name=image accept="image/*" required></label>
      <label class=drop><b>Referral document</b>
        <span class=hint>Optional &middot; image of the letter</span>
        <input type=file name=document accept="image/*"></label>
    </div>
    <div class=upfoot>
      <button type=submit>Screen this case</button>
      <span class=fine>The referral never changes the finding &mdash; it only affects whether
      the system answers. Screening runs locally on CPU and takes a few seconds.</span>
    </div>
  </form>"""


def _page(title: str, subtitle: str, body: str, upload: bool = False,
          active: str = "worklist", meta: str = "") -> str:
    def nav(key: str, href: str, label: str) -> str:
        return f'<a class="{"on" if key == active else ""}" href="{href}">{label}</a>'

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<div class=app>
<aside class=rail>
  <div class=brand><span class=mark>&#9673;</span>
    <div><b>Fundus triage</b><span>Screening support</span></div></div>
  <nav>
    <p class=railhead>Clinic</p>
    {nav("worklist", "/", "Worklist")}
    {nav("case", "#", "Case review") if active == "case" else ""}
    <a href="/#upload">Screen a new case</a>
  </nav>
  <div class=railfoot><b>Decision support only.</b>
  Every case is read by a clinician. Not a medical device.</div>
</aside>
<main>
<header>
  <div>
    <p class=eyebrow>Ophthalmic screening support</p>
    <h1>{html.escape(title)}</h1>
    <p class=sub>{html.escape(subtitle)}</p>
  </div>
  <div class=meta>{meta}</div>
</header>
<div class=wrap>{UPLOAD_PANEL if upload else ""}{body}</div>
<footer><b>{DISCLAIMER_TITLE}</b><p>{DISCLAIMER_BODY}</p></footer>
</main>
</div>
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
  <td class=conf>{case['confidence']:.0%}
    <i><b style="width:{case['confidence'] * 100:.0f}%"></b></i></td>
  <td><span class="chip {'defer' if is_defer else 'accept'}">{case['decision']}</span></td>
  <td class=reason>{html.escape(reason)}</td>
</tr>""")

    share = f"{deferred / len(cases):.0%}" if cases else "—"
    empty = ("<tr><td colspan=6 style='padding:34px 20px;text-align:center;color:var(--muted)'>"
             "The queue is empty. Screen a case with the panel above.</td></tr>")

    body = f"""
<div class=tiles>
  <div class=tile><b>{len(cases)}</b><span>cases in queue</span></div>
  <div class="tile is-defer"><b style="color:var(--defer)">{deferred}</b>
    <span>deferred for review</span><em>{share} of the queue</em></div>
  <div class="tile is-accept"><b style="color:var(--accept)">{len(cases) - deferred}</b>
    <span>accepted for reporting</span></div>
  <div class="tile is-warn"><b style="color:var(--warn)">{discordant}</b>
    <span>image–referral discordant</span></div>
</div>
<div class=tablecard>
  <div class=caption><b>Triage queue</b>
    <span>Deferred first, then ascending confidence &middot; click a case to review it</span></div>
  <table>
    <thead><tr><th>#</th><th>Case</th><th>Screening finding</th><th>Conf.</th><th>Status</th>
      <th>Reason</th></tr></thead>
    <tbody>{''.join(rows) or empty}</tbody>
  </table>
</div>
<p class=note style="margin-top:14px">The referral document never changes a finding — it only
affects whether the system answers.</p>"""

    meta = f"""<span class=pill><i></i>{len(cases)} in queue</span>
<span class=pill><i style="background:var(--defer)"></i>{deferred} deferred</span>"""

    return _page("Screening worklist",
                 "Ordered by urgency: deferred cases first, then by ascending confidence. "
                 "Queue order is what the system changes — every case is still read by a "
                 "clinician.", body, upload=True, active="worklist", meta=meta)


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
      <p class=note>{html.escape(case['image_id'])} · resized to 224×224 for the model</p>
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

    meta = f"""<span class="pill"><i style="background:var(--{'defer' if is_defer else 'accept'})">
</i>{case['decision']}</span>
<span class=pill>{case['confidence']:.0%} confidence</span>
<span class=pill>{'referral attached' if case['document_present'] else 'image only'}</span>"""

    return _page("Case review",
                 "The referral document never changes the finding. It only affects whether "
                 "the system answers.", body, active="case", meta=meta)
