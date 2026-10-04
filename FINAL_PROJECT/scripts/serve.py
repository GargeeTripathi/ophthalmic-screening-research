"""Run the screening prototype as a local web app.

    python scripts/serve.py
    python scripts/serve.py --port 8080 --cases 12

Then open http://127.0.0.1:8000

Standard library only — no Flask, no FastAPI. The app is a demonstration of the research
pipeline, and adding a web framework to it would be one more thing to install, pin and
explain for no benefit at this size.

WHAT IT DOES
------------
On startup it screens a sample of HELD-OUT cases (the deployed model's test fold, never seen
in training or model selection) with their correctly-paired referral documents, and serves the worklist. Uploading a
fundus photograph — with or without a referral — screens it live through the same pipeline
the research scripts evaluated.

Held-out matters: run this over training-fold images and every confidence reads near 100%,
the deferral rule never fires, and the demo silently misrepresents the system.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.chatbot import Chatbot  # noqa: E402
from src.app.pipeline import ScreeningPipeline  # noqa: E402
from src.app.render import case_review, worklist  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.utils.config import load_config, repo_path  # noqa: E402

UPLOAD_DIR = repo_path(load_config()["outputs"]["uploads"])
REFERRALS = repo_path("data", "sample_referrals")
MAX_UPLOAD = 12 * 1024 * 1024      # a fundus photograph; ODIR's largest is well under this
MAX_CASES = 60                     # bound the in-memory queue on a long-lived instance
CASES: list[dict] = []
PIPELINE: ScreeningPipeline | None = None
CLASS_DISPLAY: dict = {}


def decorate(result, image_path: str, document_path: str | None) -> dict:
    """Result plus everything the templates need."""
    case = result.to_dict()
    case["image_path"] = image_path
    case["document_path"] = document_path
    case["class_display"] = CLASS_DISPLAY
    case["calibration_note"] = PIPELINE.config["image_model"]["calibration_note"]
    return case


def demo_data_available() -> bool:
    """Can this machine reach the source images the demo cases need?

    A deployed copy cannot: the manifests hold absolute paths into a local ODIR checkout,
    and ODIR fundus images are not ours to redistribute. Where they are missing the app
    runs upload-only, which is the whole system minus the pre-screened queue.
    """
    sample = load_split(require_images=False).iloc[0]["image_path"]
    return sample is not None and Path(sample).is_file()


def build_demo_cases(limit: int) -> list[dict]:
    """Screen held-out images with the referral generated for that same patient and eye."""
    splits = load_split()
    held_out = splits[splits["fold"] == PIPELINE.predictor.meta["test_fold"]]
    by_image = held_out.set_index("image_id")

    documents = []
    truth = REFERRALS / "ground_truth.json"
    if truth.is_file():
        documents = json.loads(truth.read_text(encoding="utf-8"))

    # Pair each document with ITS OWN patient's image, not an arbitrary one. Random pairing
    # manufactures discordance and makes the demo look worse than the system is.
    paired, seen = [], set()
    for record in documents:
        eye_text = str(record["fields"].get("eye", "")).lower()
        eye = ("right" if eye_text in {"right", "od", "r"}
               else "left" if eye_text in {"left", "os", "l"} else None)
        if eye is None:
            continue
        image_id = f"{record['patient_id']}_{eye}"
        if image_id in by_image.index and image_id not in seen:
            seen.add(image_id)
            paired.append((by_image.loc[image_id, "image_path"], str(REFERRALS / record["file"]), image_id))

    rng = random.Random(42)
    rng.shuffle(paired)
    chosen = paired[:limit]

    # Top up with image-only cases so the demo shows both pathways.
    if len(chosen) < limit:
        for row in held_out.head(limit * 3).itertuples():
            if row.image_id not in seen:
                chosen.append((row.image_path, None, row.image_id))
                seen.add(row.image_id)
            if len(chosen) >= limit:
                break

    cases = []
    for position, (image_path, document_path, image_id) in enumerate(chosen, start=1):
        result = PIPELINE.screen(image_path, document_path)
        cases.append(decorate(result, image_path, document_path))
        print(f"  [{position}/{len(chosen)}] {image_id:<20} "
              f"{result.predicted_class}  {result.confidence:.0%}  {result.decision}")

    # Worklist order IS the product: deferred first, then least confident first.
    cases.sort(key=lambda c: (c["decision"] != "DEFER", c["confidence"]))
    return cases


def safe_name(filename: str) -> str:
    """A filename that cannot escape the upload directory or surprise the filesystem.

    Path().name alone is not enough on Linux, where a backslash is an ordinary character:
    a Windows-style path arrives as one long filename rather than being stripped.
    """
    stem = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).lstrip(".")
    return cleaned[:80] or "upload"


def parse_multipart(body: bytes, boundary: bytes) -> dict[str, tuple[str, bytes]]:
    """Minimal multipart/form-data parser — enough for two optional file fields."""
    parts = body.split(b"--" + boundary)
    files: dict[str, tuple[str, bytes]] = {}
    for part in parts:
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", "replace")
        name = re.search(r'name="([^"]+)"', headers)
        filename = re.search(r'filename="([^"]*)"', headers)
        if not name or not filename or not filename.group(1):
            continue
        files[name.group(1)] = (filename.group(1), content.rstrip(b"\r\n-"))
    return files


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):            # quiet; the startup banner is the useful output
        pass

    def _drain(self, remaining: int, chunk: int = 65536,
               ceiling: int = 64 * 1024 * 1024) -> None:
        """Discard an unwanted request body in fixed-size pieces.

        Never accumulates it, and gives up past `ceiling` rather than reading forever for a
        client that lied about, or keeps extending, its length.
        """
        if remaining < 0 or remaining > ceiling:
            self.close_connection = True
            return
        while remaining > 0:
            block = self.rfile.read(min(chunk, remaining))
            if not block:
                break
            remaining -= len(block)

    def _send(self, payload: str, status: int = 200) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802 - http.server naming
        if self.path in ("/", "/index.html"):
            self._send(worklist(CASES))
            return
        match = re.fullmatch(r"/case/(\d+)", self.path)
        if match and int(match.group(1)) < len(CASES):
            case = CASES[int(match.group(1))]
            self._send(case_review(case, Chatbot(_to_case(case)).summary()))
            return
        self._send("<h1>404</h1><p><a href='/'>back to the worklist</a></p>", 404)

    def do_POST(self):  # noqa: N802
        if self.path != "/screen":
            self._send("<h1>404</h1>", 404)
            return
        content_type = self.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            self._send("<h1>400</h1><p>expected a multipart form</p>", 400)
            return

        # Refuse oversized bodies BEFORE reading them. On localhost the declared length was
        # safe to trust; anywhere reachable it is a memory-exhaustion lever, so the read is
        # bounded whether or not the header is honest.
        try:
            declared = int(self.headers.get("Content-Length", 0))
        except ValueError:
            declared = -1
        if declared < 0 or declared > MAX_UPLOAD:
            # Answer, rather than resetting mid-upload: a client still writing its body sees
            # a connection abort instead of the reason. The body is drained in fixed chunks
            # so this stays bounded in memory - which is the thing the cap protects.
            self._drain(declared)
            self._send(f"<h1>413</h1><p>uploads are limited to "
                       f"{MAX_UPLOAD // (1024 * 1024)} MB</p>", 413)
            return

        boundary = content_type.split("boundary=")[1].strip().encode()
        body = self.rfile.read(min(declared, MAX_UPLOAD))
        files = parse_multipart(body, boundary)
        if "image" not in files:
            self._send("<h1>400</h1><p>no fundus image supplied</p>", 400)
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved = {}
        for field, (filename, content) in files.items():
            path = UPLOAD_DIR / f"upload_{field}_{safe_name(filename)}"
            path.write_bytes(content)
            saved[field] = str(path)

        try:
            result = PIPELINE.screen(saved["image"], saved.get("document"))
        except Exception:  # noqa: BLE001 - a bad upload must not take the server down
            self._send("<h1>400</h1><p>that file could not be screened as a fundus "
                       "photograph. <a href='/'>back to the worklist</a></p>", 400)
            return
        case = decorate(result, saved["image"], saved.get("document"))
        CASES.insert(0, case)
        del CASES[MAX_CASES:]        # a public instance would otherwise grow without bound
        self._send(case_review(case, Chatbot(_to_case(case)).summary()))


def _to_case(case: dict) -> dict:
    return {
        "image_id": case["image_id"], "y_pred": case["predicted_class"],
        "max_prob": case["confidence"], "agreement": case["agreement"] or 0.0,
        "stated_label": case["stated_diagnosis_class"],
        "deferred": case["decision"] == "DEFER",
        "deferral_reason": "; ".join(case["deferral_reasons"]),
        "extraction_confidence": case["extraction_confidence"],
        "document_fields": case["document_fields"],
        "calibrated": case["calibrated"],
    }


def capture() -> None:
    """Save the two screens to results/app/ as a record of what the app showed."""
    reports = repo_path(load_config()["outputs"]["results"], "app")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "worklist.html").write_text(worklist(CASES), encoding="utf-8")
    first = CASES[0]
    (reports / "case_review.html").write_text(
        case_review(first, Chatbot(_to_case(first)).summary()), encoding="utf-8")
    print(f"captured  {reports / 'worklist.html'}")
    print(f"captured  {reports / 'case_review.html'}  ({first['image_id']})")


def main() -> int:
    global PIPELINE, CLASS_DISPLAY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(__import__("os").environ.get("PORT", 8000)))
    parser.add_argument("--host", default=__import__("os").environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("OPHTHALMIC SCREENING PROTOTYPE")
    print("=" * 72)
    PIPELINE = ScreeningPipeline()
    names = PIPELINE.class_names
    CLASS_DISPLAY = {
        "N": "No referable abnormality detected",
        "D": "Signs consistent with diabetic retinopathy",
        "G": "Signs consistent with glaucoma",
        "C": "Signs consistent with cataract",
    }
    for code in names:
        CLASS_DISPLAY.setdefault(code, names[code])

    print(f"\nmodel     {PIPELINE.config['image_model']['architecture']}")
    print(f"weights   {PIPELINE.weights_source}")
    print(f"          temperature T = {PIPELINE.temperature:.3f}, calibrated on a held-out fold")
    print(f"deferral  {PIPELINE.config['deferral']['system']}, "
          f"confidence < {PIPELINE.config['deferral']['confidence_threshold']:.0%} "
          f"or agreement < {PIPELINE.config['deferral']['agreement_threshold']:.0%}")

    if args.cases and demo_data_available():
        print(f"\nscreening {args.cases} held-out cases...")
        CASES.extend(build_demo_cases(args.cases))
        deferred = sum(1 for c in CASES if c["decision"] == "DEFER")
        print(f"\n{len(CASES)} cases  ({deferred} deferred, {len(CASES) - deferred} accepted)")
        capture()
    else:
        print("\nno local ODIR checkout found - starting in UPLOAD-ONLY mode.")
        print("  the queue starts empty; upload a fundus photograph to screen a case.")
        print("  (ODIR images are not redistributed with this repository.)")

    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"
    print("\n" + "=" * 72)
    print(f"SERVING {url}    (ctrl-c to stop)")
    print("=" * 72)
    if not args.no_browser:
        webbrowser.open(url)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
