# The full screening prototype, as evaluated: image pathway, OCR, extraction, the
# reliability estimator and the deferral rules. Nothing here is a port or a reimplementation
# - it is scripts/serve.py, unchanged, which is the point of deploying it this way rather
# than rewriting the document pathway for a browser.
FROM python:3.11-slim

# opencv-python-headless still needs libGL's loader present at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000 and only $HOME is writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/user/.cache/torch \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR $HOME/app

# CPU-only torch: the GPU wheels are several GB and this runs on a CPU Space.
COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-space.txt

# Bake the ImageNet backbone into the image. src/app/pipeline.py fetches it from
# torchvision when only the 34 KB committed head is present; doing that on every cold start
# would add a ~100 MB download to the first visitor's wait.
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"

COPY --chown=user . .

EXPOSE 7860

# --cases 0 skips the demo queue: ODIR images are not redistributed, so a deployed instance
# has none to screen and starts upload-only. --no-browser because there is no browser here.
CMD ["python", "scripts/serve.py", "--cases", "0", "--no-browser"]
