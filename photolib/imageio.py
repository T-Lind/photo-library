"""Image loading helpers shared by the indexer, the API, and the models.

Centralised so that EXIF orientation, HEIC support, and the "don't decode a
45-megapixel file to make a 150px thumbnail" trick are applied identically
everywhere.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_HEIF_REGISTERED = False

# Formats Pillow reads directly (+ HEIC via pillow-heif). RAW files are
# recognised for metadata but need an embedded preview to be decodable.
RASTER_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
HEIF_EXTS = {".heic", ".heif", ".hif"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}
SUPPORTED_EXTS = RASTER_EXTS | HEIF_EXTS | RAW_EXTS

# Pillow refuses very large images by default as a decompression-bomb guard.
# Family libraries legitimately contain panoramas and scans, so raise the
# ceiling rather than dropping those photos silently.
Image.MAX_IMAGE_PIXELS = 512_000_000


def register_heif() -> None:
    """Register the HEIF/HEIC opener exactly once per process."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _HEIF_REGISTERED = True
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("pillow-heif unavailable, HEIC files will be skipped: %s", exc)


def is_supported(path: os.PathLike | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def iter_image_files(root: os.PathLike | str, follow_symlinks: bool = False) -> Iterator[Path]:
    """Yield every supported image under ``root``, recursively.

    The previous implementation used a flat ``os.listdir``, which quietly
    ignored every photo in a subfolder — i.e. almost every real library.
    Hidden directories and macOS/Windows sidecar junk are skipped.
    """
    root = Path(root)
    skip_dirs = {"@eaDir", ".thumbnails", "__pycache__"}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip_dirs]
        for name in filenames:
            if name.startswith("._") or name.startswith("."):
                continue
            p = Path(dirpath) / name
            if is_supported(p):
                yield p


def open_image(path: os.PathLike | str, target: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Open an image as RGB with EXIF orientation applied.

    ``target`` enables JPEG draft mode: libjpeg decodes straight to a reduced
    resolution, which is several times faster and uses a fraction of the
    memory when all we need is a thumbnail or a 224px model input.
    """
    register_heif()
    img = Image.open(path)
    try:
        if target is not None:
            # No-op for non-JPEG formats.
            img.draft("RGB", target)
        img = ImageOps.exif_transpose(img) or img
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
        return img
    except Exception:
        img.close()
        raise


def read_size(path: os.PathLike | str) -> Tuple[int, int]:
    """(width, height) without decoding pixel data."""
    register_heif()
    with Image.open(path) as img:
        w, h = img.size
        # Orientation 5-8 mean the stored buffer is rotated 90 degrees.
        try:
            orientation = (img.getexif() or {}).get(274, 1)
        except Exception:
            orientation = 1
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        return w, h


def to_array(img: Image.Image) -> np.ndarray:
    """RGB uint8 HxWx3 array."""
    return np.asarray(img, dtype=np.uint8)


def load_rgb_array(path: os.PathLike | str, max_side: Optional[int] = None) -> np.ndarray:
    """Load an image as an RGB array, optionally downscaled.

    Face detection does not benefit from more than ~1600px on the long edge,
    and capping it keeps peak memory bounded when a library contains 60MP
    files.
    """
    target = (max_side, max_side) if max_side else None
    with open_image(path, target=target) as img:
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        return to_array(img)
