"""Build the static site published to GitHub Pages.

Assembles docs/ from artefacts that already exist: the rendered worklist and case review,
the 14 figures, and reports/RESULTS.md. Nothing here is authored by hand, so the site cannot
drift from the experiments.

The uploader is deliberately absent. Pages serves static files, so a form that posts to
/screen would look functional and silently fail — worse than not offering it. The site says
so and points at scripts/serve.py instead.

    python scripts/build_site.py
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.render import CSS  # noqa: E402
from src.utils.config import REPO_ROOT, output_dir  # noqa: E402

FIGURE_CAPTIONS = {
    "fig01": "Acquisition source almost determines the label. Every diabetic-retinopathy "
             "image is a .jpeg and no other class contains one.",
    "fig02": "What can be predicted without looking at pathology. Metadata alone reaches "
             "66.7% against a 26.0% baseline.",
    "fig03": "Protocol ladder. No significant inflation with a linear head — a lower bound, "
             "corrected by figure 10.",
    "fig04": "Duplicate detection. ImageNet embeddings fail on fundus images; an "
             "FOV-normalised fingerprint does not.",
    "fig05": "Extraction degrades with document legibility, on layouts the extractors "
             "never saw.",
    "fig06": "Raw OCR confidence is near-useless for field correctness. A learned "
             "estimator reaches AUROC 0.98.",
    "fig07": "Risk–coverage. Document evidence roughly halves AURC; the learned gate does "
             "not beat two rules.",
    "fig08": "Triage accuracy by evidence condition and document degradation.",
    "fig09": "Dataset composition, and the age gap between classes that E11 tests.",
    "fig10": "Leakage inflation grows with the capacity to memorise. A pure retriever "
             "gains +0.045 at p ≈ 0.",
    "fig11": "Transfer to unseen acquisition sources — below majority-class guessing on "
             "three of four.",
    "fig12": "Attribution concentrates on retinal tissue at 1.56× uniform, and passes a "
             "randomisation sanity check.",
    "fig13": "The result survives age and sex matching. It is not a demographics detector.",
    "fig14": "Leakage inflation is stable across three split seeds.",
}


def markdown_to_html(source: str) -> str:
    """Just enough Markdown for RESULTS.md — headings, tables, bold, code, lists."""
    lines, out, in_table = source.split("\n"), [], False

    def inline(text: str) -> str:
        text = html.escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        return text

    for line in lines:
        stripped = line.strip()
        is_row = stripped.startswith("|") and stripped.endswith("|")

        if is_row and re.fullmatch(r"\|[\s:|-]+\|", stripped):
            continue                                    # separator row
        if is_row:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{min(level + 1, 6)}>{inline(stripped.lstrip('# '))}</h{min(level + 1, 6)}>")
        elif stripped.startswith("- "):
            out.append(f"<li>{inline(stripped[2:])}</li>")
        elif stripped:
            out.append(f"<p>{inline(stripped)}</p>")

    if in_table:
        out.append("</table>")
    return "\n".join(out)


def shell(title: str, subtitle: str, body: str, extra_css: str = "") -> str:
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}
.site{{max-width:1100px;margin:0 auto;padding:0 32px}}
nav{{display:flex;gap:18px;padding:14px 0;font-size:13px;border-bottom:1px solid var(--rule)}}
nav a{{text-decoration:none}}
figure{{margin:0 0 28px}}
figure img{{width:100%;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}}
figcaption{{font-size:12px;color:var(--muted);margin-top:8px}}
.finding{{background:var(--surface);border:1px solid var(--rule);border-left:5px solid var(--accent);
  border-radius:8px;padding:16px 20px;margin:14px 0}}
.finding h3{{margin:0 0 6px;font-size:15px}}
.finding p{{margin:0;font-size:13px;color:var(--ink-2)}}
li{{font-size:13px;color:var(--ink-2);margin:4px 0}}
h2{{margin-top:34px;font-size:20px}} h3{{font-size:15px}}
table{{margin:14px 0;font-size:12px}}
{extra_css}</style></head><body>
<header><div class=site>
  <p class=eyebrow>Ophthalmic screening support</p>
  <h1>{html.escape(title)}</h1>
  <p class=sub>{html.escape(subtitle)}</p>
</div></header>
<div class=site>
<nav><a href="./">Overview</a><a href="worklist.html">Worklist</a>
<a href="case_review.html">Case review</a><a href="figures.html">Figures</a>
<a href="results.html">Full results</a></nav>
{body}
</div>
<footer><b>Academic screening prototype. Not a medical device and not a diagnosis.</b>
<p>A qualified clinician reviews every case and makes every decision. Trained on a public
research dataset with 203 cataract and 182 glaucoma patients. Referral documents used in
development were synthetic. No prospective clinical validation.</p></footer>
</body></html>"""


def main() -> int:
    docs = REPO_ROOT / "docs"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    (docs / ".nojekyll").write_text("", encoding="utf-8")   # serve _-prefixed files verbatim

    print("=" * 72)
    print("BUILD STATIC SITE")
    print("=" * 72)

    # --- figures -------------------------------------------------------------------------
    figures = sorted(output_dir("figures", create=False).glob("*.png"))
    for figure in figures:
        shutil.copy(figure, docs / "figures" / figure.name)
    print(f"\ncopied {len(figures)} figures")

    cards = []
    for figure in figures:
        key = figure.name.split("_")[0]
        cards.append(f"""<figure><img src="figures/{figure.name}" alt="{html.escape(figure.stem)}">
<figcaption><strong>{key}</strong> — {html.escape(FIGURE_CAPTIONS.get(key, figure.stem))}</figcaption></figure>""")
    (docs / "figures.html").write_text(
        shell("Figures", "Every figure is regenerated from results/ by scripts/make_figures.py.",
              "".join(cards)), encoding="utf-8")

    # --- results -------------------------------------------------------------------------
    results_md = REPO_ROOT / "reports" / "RESULTS.md"
    if results_md.is_file():
        body = markdown_to_html(results_md.read_text(encoding="utf-8"))
        (docs / "results.html").write_text(
            shell("Results", "Generated from results/ by scripts/summarise_results.py — no "
                             "number here is transcribed by hand.", body), encoding="utf-8")
        print("rendered RESULTS.md")

    # --- the two app screens, as captured ------------------------------------------------
    for name in ("worklist.html", "case_review.html"):
        source = REPO_ROOT / "reports" / name
        if not source.is_file():
            print(f"  missing {name} - run scripts/serve.py once to generate it")
            continue
        page = source.read_text(encoding="utf-8")
        # The upload form posts to /screen, which does not exist on a static host. Leaving a
        # dead form on the page would be worse than removing it.
        page = re.sub(r"<form class=upload.*?</form>", """
  <div class="banner accept" style="background:var(--accent-soft);border-left-color:var(--accent)">
    <div><b style="color:var(--accent)">Static snapshot</b>
    <p>This is a captured page. Uploading a case needs the local server —
    <code>python scripts/serve.py</code> — because screening runs a PyTorch model that a
    static host cannot execute.</p></div>
  </div>""", page, flags=re.S)
        (docs / name).write_text(page, encoding="utf-8")
        print(f"copied {name}")

    # --- landing -------------------------------------------------------------------------
    findings = [
        ("Acquisition provenance nearly determines the label",
         "Majority baseline 26.0%. A four-row source lookup table reaches 69.1%. A "
         "metadata-only classifier reaches 66.7% without seeing a single pixel. Within one "
         "source the signal vanishes (+0.004)."),
        ("Patient leakage is real, confirmed on a deployable model",
         "A linear head gains +0.005 (p=0.75) — it cannot memorise. A fine-tuned ResNet50 "
         "on a GPU gains +0.022 at p<0.001; a 1-NN retriever gains +0.045 at p≈0. Splitting "
         "by patient matters more as model capacity rises."),
        ("The model does not transfer across acquisition sources",
         "Trained on ODIR, it scores below majority-class guessing on three of four unseen "
         "sources."),
        ("It is not a demographics detector",
         "Cataract-vs-rest AUROC holds near 0.94 under age and sex matching, and age plus "
         "sex alone score below the majority baseline."),
        ("Raw OCR confidence is near-useless for clinical field extraction",
         "AUROC 0.52 and 0.64. A learned correctness estimator reaches 0.98 and 0.99; the "
         "signal is in value format and length, not in OCR confidence."),
        ("Two negative results are reported, not tuned away",
         "The protocol ladder's initial null, later corrected by the capacity sweep; and the "
         "learned deferral gate, which lost to two hand-written rules and was dropped by a "
         "kill rule written before the run."),
    ]
    body = f"""
<div class=finding style="border-left-color:var(--defer)">
  <h3>What this project asks</h3>
  <p>Public four-class fundus benchmarks are widely reported at 95–96% accuracy. This asks
  how much of that is retinal pathology and how much is <em>where the image came from</em>,
  re-evaluates standard classifiers under a corrected protocol, and builds a
  human-in-the-loop screening prototype around the answer.</p>
</div>

<h2>Findings</h2>
{''.join(f'<div class=finding><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p></div>'
         for t, d in findings)}

<h2>The prototype</h2>
<p><strong>The referral document never changes the diagnosis.</strong> That is enforced in
code — <code>predict_image()</code> does not take the document as a parameter — and the
interface mirrors it: the image pathway and the document pathway are separate columns that
meet only at the decision banner. The referral, and how reliably it was read, decide only
<em>whether</em> the system answers.</p>
<ul>
  <li><a href="worklist.html">Worklist</a> — the triage queue, deferred cases first</li>
  <li><a href="case_review.html">Case review</a> — one case end to end, with the grounded assistant</li>
  <li><a href="figures.html">Figures</a> — all {len(figures)}, regenerated from results/</li>
  <li><a href="results.html">Full results</a> — generated from results/, nothing transcribed</li>
</ul>

<h2>Running it</h2>
<p>The live app needs a server, because screening runs a PyTorch model:</p>
<ul>
  <li><code>pip install -r requirements.txt &amp;&amp; pip install -e .</code></li>
  <li><code>python scripts/serve.py</code> → <code>http://127.0.0.1:8000</code></li>
  <li><code>python scripts/run_all.py</code> rebuilds every experiment (~70 min, CPU)</li>
</ul>
<p>The repository is self-contained: the committed model is 34 KB, and the ImageNet backbone
is fetched from torchvision at runtime. ODIR images are not redistributed, so a fresh clone
starts in upload-only mode.</p>

<h2>Honest limits</h2>
<ul>
  <li>203 cataract and 182 glaucoma patients — intervals, not point estimates.</li>
  <li>Every model sits on frozen ImageNet features; a fine-tuned backbone may differ.</li>
  <li>The referral corpus is synthetic. Technical evaluation, never clinical validation.</li>
  <li>Attribution shows where attribution falls, not that the model sees a lesion.</li>
  <li>No prospective data, no clinical trial, no regulatory clearance.</li>
</ul>"""

    (docs / "index.html").write_text(
        shell("Provenance-aware ophthalmic screening",
              "An empirical audit of a widely used fundus benchmark, and a "
              "human-in-the-loop screening prototype built around what it found.",
              body), encoding="utf-8")

    written = sorted(p.name for p in docs.glob("*.html"))
    print(f"\nwritten to docs/: {', '.join(written)}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def build_standalone() -> Path:
    """Emit docs/standalone.html - the whole site as one portable file.

    GitHub Pages refuses private repositories on a free plan, so the site also has to be
    publishable somewhere that does not need a host. Everything is inlined: images as data
    URIs, both app screens, and the results table.
    """
    import base64

    docs = REPO_ROOT / "docs"
    figures = sorted((docs / "figures").glob("*.png"))

    cards = []
    for figure in figures:
        key = figure.name.split("_")[0]
        encoded = base64.b64encode(figure.read_bytes()).decode()
        cards.append(
            f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(figure.stem)}">'
            f'<figcaption><strong>{key}</strong> — '
            f'{html.escape(FIGURE_CAPTIONS.get(key, figure.stem))}</figcaption></figure>')

    def inner(name: str) -> str:
        path = docs / name
        if not path.is_file():
            return "<p>not generated</p>"
        page = path.read_text(encoding="utf-8")
        match = re.search(r"<div class=wrap>(.*?)</div>\s*<footer", page, re.S)
        return match.group(1) if match else ""

    results = ""
    results_md = REPO_ROOT / "reports" / "RESULTS.md"
    if results_md.is_file():
        results = markdown_to_html(results_md.read_text(encoding="utf-8"))

    index = (docs / "index.html").read_text(encoding="utf-8")
    overview = re.search(r"<nav>.*?</nav>(.*?)</div>\s*<footer", index, re.S)
    overview_html = overview.group(1) if overview else ""
    overview_html = re.sub(r'<a href="(worklist|case_review|figures|results)\.html">',
                           lambda m: f'<a href="#{m.group(1)}">', overview_html)

    sections = [("overview", "Overview", overview_html),
                ("worklist", "Worklist", inner("worklist.html")),
                ("case_review", "Case review", inner("case_review.html")),
                ("figures", "Figures", "".join(cards)),
                ("results", "Full results", results)]

    tabs = "".join(f'<button data-tab="{i}" class="{"on" if n == 0 else ""}">'
                   f'{html.escape(label)}</button>'
                   for n, (i, label, _) in enumerate(sections))
    panels = "".join(f'<section id="{i}" class=panel {"" if n == 0 else "hidden"}>{content}</section>'
                     for n, (i, label, content) in enumerate(sections))

    body = f"""<div class=tabs>{tabs}</div>{panels}
<script>
document.querySelectorAll('.tabs button').forEach(function (button) {{
  button.addEventListener('click', function () {{
    document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('on'));
    document.querySelectorAll('.panel').forEach(p => p.hidden = true);
    button.classList.add('on');
    document.getElementById(button.dataset.tab).hidden = false;
    window.scrollTo(0, 0);
  }});
}});
document.querySelectorAll('a[href^="#"]').forEach(function (link) {{
  link.addEventListener('click', function (event) {{
    event.preventDefault();
    var target = link.getAttribute('href').slice(1);
    var button = document.querySelector('.tabs button[data-tab="' + target + '"]');
    if (button) button.click();
  }});
}});
</script>"""

    extra = """
.tabs{display:flex;gap:6px;flex-wrap:wrap;padding:14px 0;border-bottom:1px solid var(--rule)}
.tabs button{background:transparent;color:var(--ink-2);border:1px solid var(--rule);
  border-radius:99px;padding:7px 15px;font-size:12.5px;font-weight:500;cursor:pointer}
.tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tabs button:hover:not(.on){background:var(--surface-2)}
.panel{padding-top:18px}
.panel[hidden]{display:none}
"""
    page = shell("Provenance-aware ophthalmic screening",
                 "An empirical audit of a widely used fundus benchmark, and a "
                 "human-in-the-loop screening prototype built around what it found.",
                 body, extra_css=extra)
    page = page.replace("<nav>", "<nav hidden>")

    out = docs / "standalone.html"
    out.write_text(page, encoding="utf-8")
    return out
