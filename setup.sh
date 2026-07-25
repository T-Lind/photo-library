#!/usr/bin/env bash
# Set up photolib from a clean machine.
#
# Works on Linux and macOS. Unlike the previous version this needs no cmake
# and no dlib compilation — the face models run through ONNX Runtime, which
# ships prebuilt wheels, so installation is just pip.
set -euo pipefail

PYTHON=${PYTHON:-python3}
GPU=${GPU:-auto}   # auto | cuda | cpu

echo "==> Checking Python"
$PYTHON -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || {
    echo "Python 3.10 or newer is required." >&2
    exit 1
}

if [ ! -d .venv ]; then
    echo "==> Creating virtualenv"
    $PYTHON -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

if [ "$GPU" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then GPU=cuda; else GPU=cpu; fi
fi

echo "==> Installing PyTorch ($GPU)"
if [ "$GPU" = "cpu" ]; then
    # The default PyPI torch wheel drags in ~2.5 GB of CUDA libraries that a
    # CPU-only machine will never use.
    pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch
fi

echo "==> Installing photolib"
pip install -r requirements.txt

if [ "$GPU" = "cuda" ]; then
    echo "==> Switching to GPU ONNX Runtime for face recognition"
    pip uninstall -y onnxruntime >/dev/null 2>&1 || true
    pip install onnxruntime-gpu
fi

cat <<'EOF'

Done. Next:

    source .venv/bin/activate
    python -m photolib.cli index ~/Pictures      # first index (downloads models)
    python run.py                                # start the API on :8000

The web UI lives in the photo-library-frontend repository.
EOF
