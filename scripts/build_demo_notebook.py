"""Generate colab/live_demo.ipynb - the full prototype, live, on a shareable URL.

    python scripts/build_demo_notebook.py

WHY THIS EXISTS
---------------
The deployed site screens an image in the browser but cannot read a referral: OCR, the
layout-aware extraction and the reliability estimator are Python. Hosting the real thing
needs a machine that runs Python, and the free options are poor - Hugging Face now bills
Docker Spaces, and Render's free tier is 512 MB and 0.1 CPU. Colab is free, already part of
this project's workflow, and fast enough to actually demonstrate the system.

The trade is that the URL lives only while the notebook runs, which is right for a
demonstration and wrong for a permanent link.

WHAT IT PRODUCES
----------------
A self-contained notebook: the ~0.3 MB runtime bundle is embedded as base64, so there is no
upload step and no clone of a private repository. Open it in Colab, run all, share the
printed URL. The tunnel is a Cloudflare quick tunnel, which needs no account.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import REPO_ROOT  # noqa: E402

NOTEBOOK = REPO_ROOT / "colab" / "live_demo.ipynb"
PORT = 7860


def build_bundle() -> bytes:
    """Reuse build_space.py's tree - the same files the app needs anywhere."""
    tree = REPO_ROOT / "build" / "space"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_space.py")],
                   cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(tree.rglob("*")):
            if path.is_file() and path.name not in ("Dockerfile", "requirements-space.txt"):
                archive.write(path, path.relative_to(tree))
    return buffer.getvalue()


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main() -> int:
    print("=" * 72)
    print("BUILD LIVE DEMO NOTEBOOK")
    print("=" * 72)

    payload = base64.b64encode(build_bundle()).decode()
    print(f"\nbundle embedded: {len(payload) / 1024:.0f} KB of base64")

    cells = [
        markdown("""# Ophthalmic screening prototype - live demo

**Academic screening prototype. Not a medical device and not a diagnosis.** A qualified
clinician reviews every case and makes every decision. Trained on a public research dataset
with 203 cataract and 182 glaucoma patients; the referral documents used in development were
synthetic; there has been no prospective clinical validation.

This runs the **complete system** - the image model, OCR, layout-aware field extraction, the
learned reliability estimator, the concordance check and the deferral rules - and puts it on
a public URL you can share.

Nothing here is reimplemented for the web. It is the same code that produced the reported
results.

**To run it:** `Runtime -> Run all`, then open the URL printed by the last cell. Upload a
fundus photograph, and a referral letter image if you have one.

**The URL lives only while this notebook is running.** Closing the tab or letting Colab time
out takes the demo down with it. That is the trade for it being free and full-speed.
"""),
        markdown("## 1. Unpack the prototype\n\nThe runtime bundle is embedded in this "
                 "notebook, so there is nothing to upload and no repository to clone."),
        code(f'''import base64, io, zipfile, pathlib

BUNDLE = "{payload}"

root = pathlib.Path("/content/screening")
if root.exists():
    import shutil; shutil.rmtree(root)
root.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
    archive.extractall(root)

print("unpacked to", root)
for path in sorted(root.rglob("*")):
    if path.is_file() and path.suffix in (".pt", ".joblib", ".json", ".py") \\
            and "checkpoints" in str(path):
        print(f"  {{path.stat().st_size / 1024:8.0f}} KB  {{path.relative_to(root)}}")
'''),
        markdown("## 2. Dependencies\n\nColab already has torch, torchvision, OpenCV, "
                 "scikit-learn, pandas and Pillow. Only the OCR stack is missing."),
        code('''%pip install -q rapidocr-onnxruntime rapidfuzz

import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
'''),
        markdown("## 3. Start the app and open a public URL\n\nThe server runs in the "
                 "background. The tunnel is a Cloudflare quick tunnel - no account, no "
                 "signup."),
        code(f'''import subprocess, time, re, threading, pathlib, urllib.request

ROOT = pathlib.Path("/content/screening")
PORT = {PORT}

# ODIR images are not redistributed, so there is no pre-screened queue: --cases 0 starts the
# app upload-only, which is the mode a deployed instance runs in anyway.
server = subprocess.Popen(
    ["python", "scripts/serve.py", "--cases", "0", "--no-browser",
     "--port", str(PORT), "--host", "127.0.0.1"],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def echo():
    for line in server.stdout:
        print("[app]", line.rstrip())
threading.Thread(target=echo, daemon=True).start()

for _ in range(120):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{{PORT}}/", timeout=3)
        print("\\napp is up")
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("the app did not start - see the [app] output above")

!wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /content/cloudflared

tunnel = subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{{PORT}}",
     "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# cloudflared keeps logging. Its output must be drained for the whole session: stopping at
# the URL leaves the pipe to fill, and a full pipe blocks the tunnel process.
found = {{}}
def watch():
    for line in tunnel.stdout:
        hit = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", line)
        if hit and "url" not in found:
            found["url"] = hit.group(0)
threading.Thread(target=watch, daemon=True).start()

for _ in range(60):
    if "url" in found:
        break
    time.sleep(1)

public = found.get("url")
if not public:
    raise SystemExit("the tunnel did not report a URL - re-run this cell")

# Prove it end to end, through Cloudflare, before handing the link over.
time.sleep(3)
try:
    check = urllib.request.urlopen(urllib.request.Request(
        public, headers={{"User-Agent": "Mozilla/5.0"}}), timeout=60).status
except Exception as exc:
    check = f"unreachable ({{type(exc).__name__}})"

print("\\n" + "=" * 68)
print("  SHARE THIS URL:", public)
print("  reachable from outside:", check, "(200 means it is genuinely live)")
print("=" * 68)
print("\\n  Upload a fundus photograph, and a referral letter image if you have one.")
print("  The URL stays live only while this notebook is running - leave the tab open.")
'''),
        markdown("""## 4. While it runs

Leave this notebook open. Colab disconnects idle sessions, which takes the URL down with it.

**What to try.** Upload a fundus photograph on its own and the case is judged on image
confidence alone. Upload one *with* a referral and the document pathway runs too: the fields
are read, their reliability is estimated, and the stated diagnosis is compared against the
image finding. A case is deferred when confidence falls below 60%, when the referral
contradicts the image, or when the fields were read unreliably.

The referral never changes the finding. It only changes whether the system answers.

**To stop:** `Runtime -> Disconnect and delete runtime`, or set the flag in the cell below
and run it.

The flag exists because this notebook is meant to be run with `Run all`, and a teardown cell
that runs unguarded would take the demo down a second after printing its URL."""),
        code('''# Set this to True and run the cell to take the demo down.
STOP = False

if STOP:
    tunnel.terminate(); server.terminate()
    print("demo stopped - the public URL is now dead")
else:
    alive = server.poll() is None and tunnel.poll() is None
    print("demo is still running" if alive else "demo is NOT running - re-run the cell above")
    print("set STOP = True and re-run this cell to stop it")
'''),
    ]

    notebook = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }

    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"written {NOTEBOOK}  ({NOTEBOOK.stat().st_size / 1024:.0f} KB)")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
