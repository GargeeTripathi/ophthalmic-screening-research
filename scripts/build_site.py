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
import json
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


def browser_panel(labels: list[str], display: dict, threshold: float, note: str) -> str:
    """The intake panel, re-implemented to run the model in the visitor's browser.

    A static host cannot execute PyTorch, so the deployed page used to replace the panel
    with a notice. It does not have to: scripts/export_model.py fuses the whole image
    pathway - backbone, L2 normalisation, head, temperature, softmax - into one ONNX graph,
    verified against PyTorch at export time, and onnxruntime-web runs it here.

    TWO HONEST LIMITS, both stated on the page rather than hidden:

    1. Image only. The referral pathway is RapidOCR under Python and has no browser
       equivalent, so this cannot reproduce the deferral rules that need a document. Only
       the confidence rule applies.
    2. Preprocessing had to be rebuilt, not borrowed. Canvas drawImage area-averages, which
       on a 10x downscale moved confidence by ~3 points - enough to cross the deferral
       threshold - so Pillow's bilinear filter is reimplemented in the page. Measured
       agreement after that is ~7e-4 against the PyTorch pipeline.
    """
    names = json.dumps([display.get(code, code) for code in labels])
    return f"""
  <form class=upload id=upload onsubmit="return false">
    <div class=uphead>
      <span class=upmark>+</span>
      <div><b>Screen a new case</b>
      <p>Runs the real screening model in your browser - the same fused graph the local app
      evaluates, so the finding matches. Nothing is uploaded anywhere; the image never leaves
      this device. The model is ~90&nbsp;MB and loads once.</p></div>
    </div>
    <div class=drops>
      <label class="drop req"><b>Fundus photograph</b>
        <span class=hint>Required &middot; JPG or PNG</span>
        <input type=file id=shot accept="image/*"></label>
      <label class=drop style="border-style:solid;opacity:.62"><b>Referral document</b>
        <span class=hint>Unavailable in the browser</span>
        <p class=note style="margin:9px 0 0">Reading a referral needs the OCR stack, which
        runs under Python. Use <code>scripts/serve.py</code> for the document pathway.</p>
      </label>
    </div>
    <div class=upfoot>
      <button type=button id=go>Screen this case</button>
      <span class=fine id=state>Only the confidence rule applies here, because no referral
      can be read. The referral never changes the finding in any case.</span>
    </div>
    <div id=out style="margin-top:16px"></div>
  </form>
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js"></script>
  <script>
  (function () {{
    var LABELS = {json.dumps(labels)}, NAMES = {names}, THRESHOLD = {threshold};
    var MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
    var session = null, busy = false;
    var shot = document.getElementById('shot'), go = document.getElementById('go');
    var state = document.getElementById('state'), out = document.getElementById('out');

    function say(text) {{ state.textContent = text; }}

    // Pillow's BILINEAR resize, reimplemented, because canvas drawImage does area-averaging
    // instead. On a 10x downscale that disagreed with the pipeline by ~3 points of
    // confidence - enough to move a case across the 60% deferral threshold. Pillow uses a
    // separable triangle filter whose support scales with the reduction factor, and rounds
    // to 8-bit between the two passes; both matter. Measured against the PyTorch pipeline
    // after this: 0.6755 vs 0.6762 on a 2304x1728 fundus photograph.
    function clamp255(v) {{ v = Math.round(v); return v < 0 ? 0 : v > 255 ? 255 : v; }}

    function weights(inSize, outSize) {{
      var scale = inSize / outSize, support = scale > 1 ? scale : 1, rows = [];
      for (var i = 0; i < outSize; i++) {{
        var centre = (i + 0.5) * scale;
        var lo = Math.max(0, Math.floor(centre - support + 0.5));
        var hi = Math.min(inSize, Math.ceil(centre + support + 0.5));
        var w = [], total = 0;
        for (var x = lo; x < hi; x++) {{
          var t = Math.abs((x + 0.5 - centre) / support), v = t < 1 ? 1 - t : 0;
          w.push(v); total += v;
        }}
        for (var k = 0; k < w.length; k++) w[k] /= total || 1;
        rows.push({{min: lo, w: w}});
      }}
      return rows;
    }}

    function resize(px, inW, inH, out) {{
      var hx = weights(inW, out), mid = new Float32Array(inH * out * 3);
      for (var y = 0; y < inH; y++)
        for (var i = 0; i < out; i++) {{
          var r = hx[i], a = [0, 0, 0];
          for (var k = 0; k < r.w.length; k++) {{
            var o = (y * inW + r.min + k) * 4, wt = r.w[k];
            a[0] += px[o] * wt; a[1] += px[o + 1] * wt; a[2] += px[o + 2] * wt;
          }}
          var m = (y * out + i) * 3;
          mid[m] = clamp255(a[0]); mid[m + 1] = clamp255(a[1]); mid[m + 2] = clamp255(a[2]);
        }}
      var vy = weights(inH, out), res = new Float32Array(out * out * 3);
      for (var j = 0; j < out; j++) {{
        var rr = vy[j];
        for (var x2 = 0; x2 < out; x2++) {{
          var b = [0, 0, 0];
          for (var k2 = 0; k2 < rr.w.length; k2++) {{
            var o2 = ((rr.min + k2) * out + x2) * 3, w2 = rr.w[k2];
            b[0] += mid[o2] * w2; b[1] += mid[o2 + 1] * w2; b[2] += mid[o2 + 2] * w2;
          }}
          var q = (j * out + x2) * 3;
          res[q] = b[0]; res[q + 1] = b[1]; res[q + 2] = b[2];
        }}
      }}
      return res;
    }}

    // Must match src/app/pipeline.py: resize to 224 square, ImageNet mean/std, NCHW.
    // The L2 normalisation and temperature are inside the graph, not here.
    function tensorFrom(img) {{
      var c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      var g = c.getContext('2d', {{willReadFrequently: true}});
      g.drawImage(img, 0, 0);
      var px = g.getImageData(0, 0, c.width, c.height).data;
      var rgb = resize(px, c.width, c.height, 224);
      var data = new Float32Array(3 * 224 * 224);
      for (var i = 0; i < 224 * 224; i++) {{
        for (var ch = 0; ch < 3; ch++) {{
          data[ch * 224 * 224 + i] = (clamp255(rgb[i * 3 + ch]) / 255 - MEAN[ch]) / STD[ch];
        }}
      }}
      return new ort.Tensor('float32', data, [1, 3, 224, 224]);
    }}

    function render(probs, url) {{
      var order = probs.map(function (p, i) {{ return {{p: p, i: i}}; }})
                       .sort(function (a, b) {{ return b.p - a.p; }});
      var top = order[0], defer = top.p < THRESHOLD;
      var bars = order.map(function (o) {{
        return '<div class=bar><span>' + NAMES[o.i] + '</span>' +
               '<span class=track><i class="' + (o === top ? '' : 'dim') +
               '" style="width:' + (o.p * 100).toFixed(0) + '%"></i></span>' +
               '<span style="text-align:right">' + (o.p * 100).toFixed(0) + '%</span></div>';
      }}).join('');
      out.innerHTML =
        '<div class="banner ' + (defer ? '' : 'accept') + '">' +
          '<span class="chip ' + (defer ? 'defer' : 'accept') + '">' +
          (defer ? 'DEFER' : 'ACCEPT') + '</span><div><b>' +
          (defer ? 'Referred to clinician' : 'Accepted for reporting') + '</b><p>' +
          (defer ? 'Image confidence ' + (top.p * 100).toFixed(0) + '% is below the ' +
                   (THRESHOLD * 100).toFixed(0) + '% threshold.'
                 : 'Image confidence is above the threshold. No referral was read, so the ' +
                   'concordance rules did not run.') +
          '</p></div></div>' +
        '<div class=cols><div class=card><p class=label>Fundus photograph</p>' +
          '<img class=fundus src="' + url + '" alt="uploaded fundus photograph">' +
          '<p class=note>screened locally in your browser, 224&times;224</p></div>' +
        '<div class=card><p class=label>Screening finding &middot; image only</p>' +
          '<p class=finding>' + NAMES[top.i] + '</p>' +
          '<p class=conf>' + (top.p * 100).toFixed(0) + '%</p>' +
          '<p class=note>{html.escape(note)}</p>' +
          '<p class=label style="margin-top:16px">Class probabilities</p>' + bars +
          '</div></div>';
    }}

    go.addEventListener('click', function () {{
      var file = shot.files && shot.files[0];
      if (!file) {{ say('Choose a fundus photograph first.'); return; }}
      if (busy) return;
      busy = true;
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () {{
        (session ? Promise.resolve(session)
                 : (say('Loading the model (~90 MB, once)...'),
                    ort.InferenceSession.create('model/screening_fused.onnx',
                      {{executionProviders: ['wasm']}})))
          .then(function (s) {{
            session = s;
            say('Screening...');
            return s.run({{image: tensorFrom(img)}});
          }})
          .then(function (result) {{
            render(Array.from(result.probabilities.data), url);
            say('Screened in your browser, image only. Preprocessing reproduces the '
                + 'pipeline\\'s resampling, so confidences track the local app to about '
                + '0.001; the displayed percentage is the same.');
            busy = false;
          }})
          .catch(function (err) {{
            say('Could not screen this image: ' + err.message);
            busy = false;
          }});
      }};
      img.onerror = function () {{ say('That file could not be read as an image.');
                                   busy = false; }};
      img.src = url;
    }});
  }})();
  </script>"""


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


DISPLAY_NAMES = {
    "N": "No referable abnormality detected",
    "D": "Signs consistent with diabetic retinopathy",
    "G": "Signs consistent with glaucoma",
    "C": "Signs consistent with cataract",
}

STATIC_NOTICE = """
  <div class="banner accept" style="background:var(--accent-soft);border-left-color:var(--accent)">
    <div><b style="color:var(--accent)">Static snapshot</b>
    <p>This is a captured page. Uploading a case needs the local server &mdash;
    <code>python scripts/serve.py</code> &mdash; because screening runs a model that a
    static host cannot execute.</p></div>
  </div>"""

PAGES = (("./", "index", "Overview"), ("worklist.html", "worklist", "Worklist"),
         ("case_review.html", "case", "Case review"), ("figures.html", "figures", "Figures"),
         ("results.html", "results", "Full results"))


def shell(title: str, subtitle: str, body: str, extra_css: str = "",
          active: str = "index") -> str:
    """Wrap page content in the dashboard frame — the same rail the live app uses.

    `<div class="site wrap" id=main>` is load-bearing: build_standalone() extracts page
    content by that marker, so it must stay a single element closing right before <footer>.
    """
    nav = "".join(f'<a class="{"on" if key == active else ""}" href="{href}">{label}</a>'
                  for href, key, label in PAGES)
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}
.site{{max-width:1180px}}
figure{{margin:0 0 28px}}
figure img{{width:100%;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}}
figcaption{{font-size:12px;color:var(--muted);margin-top:8px}}
.finding{{background:var(--surface);border:1px solid var(--rule);border-left:5px solid var(--accent);
  border-radius:10px;padding:16px 20px;margin:14px 0;box-shadow:var(--shadow)}}
.finding h3{{margin:0 0 6px;font-size:15px}}
.finding p{{margin:0;font-size:13px;color:var(--ink-2)}}
li{{font-size:13px;color:var(--ink-2);margin:4px 0}}
h2{{margin-top:34px;font-size:20px}} h3{{font-size:15px}}
table{{margin:14px 0;font-size:12px;border:1px solid var(--rule);border-radius:10px;
  overflow:hidden}}
{extra_css}</style></head><body>
<div class=app>
<aside class=rail>
  <div class=brand><span class=mark>&#9673;</span>
    <div><b>Fundus triage</b><span>Screening support</span></div></div>
  <nav><p class=railhead>Project</p>{nav}</nav>
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
</header>
<div class="site wrap" id=main>{body}</div>
<footer><b>Academic screening prototype. Not a medical device and not a diagnosis.</b>
<p>A qualified clinician reviews every case and makes every decision. Trained on a public
research dataset with 203 cataract and 182 glaucoma patients. Referral documents used in
development were synthetic. No prospective clinical validation.</p></footer>
</main>
</div>
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
              "".join(cards), active="figures"), encoding="utf-8")

    # --- results -------------------------------------------------------------------------
    results_md = REPO_ROOT / "reports" / "RESULTS.md"
    if results_md.is_file():
        body = markdown_to_html(results_md.read_text(encoding="utf-8"))
        (docs / "results.html").write_text(
            shell("Results", "Generated from results/ by scripts/summarise_results.py — no "
                             "number here is transcribed by hand.", body, active="results"),
            encoding="utf-8")
        print("rendered RESULTS.md")

    # --- the fused model, so the deployed panel can actually screen ----------------------
    fused = output_dir("checkpoints", create=False) / "screening_fused.onnx"
    replacement = STATIC_NOTICE
    if fused.is_file():
        (docs / "model").mkdir(parents=True, exist_ok=True)
        shutil.copy(fused, docs / "model" / fused.name)
        deployment = json.loads(
            (fused.parent / "deployment.json").read_text(encoding="utf-8"))
        replacement = browser_panel(
            deployment["labels"], DISPLAY_NAMES,
            deployment["deferral"]["confidence_threshold"],
            deployment["image_model"]["calibration_note"])
        print(f"copied {fused.name}  ({fused.stat().st_size / 1e6:.0f} MB) "
              "- the deployed panel screens in-browser")
    else:
        print("no screening_fused.onnx - deployed panel falls back to the static notice")
        print("  (run scripts/export_model.py to build it)")

    # --- the two app screens, as captured ------------------------------------------------
    for name in ("worklist.html", "case_review.html"):
        source = REPO_ROOT / "reports" / name
        if not source.is_file():
            print(f"  missing {name} - run scripts/serve.py once to generate it")
            continue
        page = source.read_text(encoding="utf-8")
        # The captured form posts to /screen, which no static host provides. Where the fused
        # ONNX is available the panel is replaced by one that runs the model in the browser;
        # where it is not, by a notice. Never by a form that would fail silently.
        page = re.sub(r"<form class=upload.*?</form>", replacement, page, flags=re.S)
        # Every other app route is equally absent here. Rail links move to their static
        # equivalents; per-case links become plain text, because linking all ten rows to the
        # one captured review would show the wrong case for nine of them.
        page = re.sub(r'<a class="[^"]*" href="/">', '<a class="" href="./">', page)
        page = page.replace('<a href="/#upload">Screen a new case</a>',
                            '<a href="case_review.html">Case review</a>')
        page = page.replace('href="/"', 'href="./"')
        page = re.sub(r'<a href="/case/\d+">(.*?)</a>', r"\1", page)
        page = page.replace("click a case to review it",
                            "see Case review for one case end to end")
        (docs / name).write_text(page, encoding="utf-8")
        print(f"copied {name}")

    # --- landing -------------------------------------------------------------------------
    findings = [
        ("Acquisition provenance nearly determines the label",
         "Majority baseline 26.0%. A four-row source lookup table reaches 69.1%. A "
         "metadata-only classifier reaches 66.7% without seeing a single pixel. Holding "
         "source constant removes the signal in odir_jpg512 (+0.004) but not in crop_jpg256 "
         "(+0.219) or png_highres (+0.284): provenance explains the diabetic-retinopathy and "
         "normal classes almost entirely, and contributes little to separating cataract from "
         "glaucoma. Both halves are reported."),
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
    overview = re.search(r'<div class="site wrap" id=main>(.*?)</div>\s*<footer', index, re.S)
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
