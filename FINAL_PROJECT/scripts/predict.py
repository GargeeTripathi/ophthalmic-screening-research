"""Classify one fundus photograph with the final ConvNeXt-Tiny model.

    python scripts/predict.py --image path/to/fundus.jpg
    python scripts/predict.py --image path/to/fundus.jpg --gradcam out.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.predict import Predictor  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--gradcam", type=Path, default=None, help="also save a Grad-CAM overlay to this file")
    args = p.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"no such image: {args.image}")

    predictor = Predictor()
    result = predictor.predict(args.image)
    if args.gradcam:
        from PIL import Image
        from src.data.preprocessing import load_resized, to_tensor
        from src.explainability.gradcam import grad_cam, overlay
        pixels = load_resized(args.image)
        cam, _, _ = grad_cam(predictor.model, to_tensor(pixels))
        args.gradcam.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay(pixels, cam)).save(args.gradcam)
        result["gradcam"] = str(args.gradcam)
    print(json.dumps(result, indent=2))
    print(f"\n{result['predicted_name']}  ({result['confidence']:.1%} confidence)  -  "
          "academic prototype, not a diagnosis")
