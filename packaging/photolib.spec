# PyInstaller spec for the photolib sidecar binary.
#
#   pyinstaller packaging/photolib.spec --noconfirm
#
# Produces dist/photolib-server/ containing the executable, the exported
# ONNX model, and the built web UI. Everything the desktop app needs to run
# is in that folder — no Python, no Node.js, no PyTorch.
#
# One-folder rather than one-file on purpose: --onefile unpacks ~500 MB to a
# temp directory on every launch, which turns a two-second start into twenty.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
MODEL_DIR = Path(os.environ.get("PHOTOLIB_MODEL_DIR", ROOT / "models" / "siglip2-base"))
WEB_DIR = Path(os.environ.get("PHOTOLIB_WEB_DIR", ROOT / "desktop" / "ui"))

datas = []

MODEL_FILES = (
    "text.onnx",
    "vision.onnx",
    "tokenizer.json",
    "preprocess.json",
    "golden.json",
)
missing_model_files = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
if missing_model_files:
    print(f"WARNING: incomplete exported model at {MODEL_DIR}; missing "
          f"{', '.join(missing_model_files)}", file=sys.stderr)
else:
    # List the runtime assets explicitly. Copying the whole model directory can
    # accidentally ship optional quantized/export scratch files and push the
    # Windows release over GitHub's 2 GiB asset limit.
    datas += [
        (str(MODEL_DIR / name), "models/siglip2-base")
        for name in MODEL_FILES
    ]

if WEB_DIR.is_dir():
    datas.append((str(WEB_DIR), "web"))
else:
    print(f"WARNING: no built web UI at {WEB_DIR}; "
          "desktop/ui must contain index.html", file=sys.stderr)

# onnxruntime and lancedb ship compiled extensions and data files that
# PyInstaller's static analysis does not find on its own.
binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("lance")
datas += collect_data_files("onnxruntime")

# RapidOCR keeps its detection/recognition models and config inside the
# package; collect them so text-in-photos search works when frozen. The
# try/except keeps the spec usable in a build without OCR.
try:
    datas += collect_data_files("rapidocr_onnxruntime")
except Exception as exc:
    print(f"WARNING: rapidocr not bundled ({exc}); "
          "text-in-photos search will be unavailable", file=sys.stderr)

# imageio-ffmpeg carries the ffmpeg executable as package data — that binary
# is what indexes and poster-frames videos. Without it, videos are skipped.
try:
    datas += collect_data_files("imageio_ffmpeg")
except Exception as exc:
    print(f"WARNING: imageio-ffmpeg not bundled ({exc}); "
          "videos will not be indexed", file=sys.stderr)

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "photolib.embeddings.onnx_vision",
    "photolib.faces.insight",
    "pillow_heif",
    "tokenizers",
    "rapidocr_onnxruntime",
    # Native Windows OCR projection modules (loaded dynamically).
    "winsdk.windows.media.ocr",
    "winsdk.windows.graphics.imaging",
    "winsdk.windows.storage.streams",
    # Recycle Bin support; the platform backend is chosen at runtime.
    "send2trash",
    "send2trash.win",
    "send2trash.win.modern",
    # Video indexing and playback posters.
    "imageio_ffmpeg",
]

# Nothing here uses PyTorch — the ONNX backend is the entire point. Excluding
# them explicitly keeps a stray transitive import from dragging in gigabytes.
excludes = [
    "torch", "torchvision", "transformers", "tensorflow", "jax",
    "matplotlib", "pandas", "IPython", "notebook", "pytest",
    "tkinter", "sklearn",
]

a = Analysis(
    [str(ROOT / "photolib" / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="photolib-server",
    debug=False,
    strip=False,
    upx=False,          # UPX-compressed binaries are a common false positive
    console=True,       # stdout is the readiness handshake with the shell
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="photolib-server",
)
