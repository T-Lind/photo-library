"""Image and video loading helpers shared by the indexer, the API, and models.

Centralised so that EXIF orientation, HEIC support, and the "don't decode a
45-megapixel file to make a 150px thumbnail" trick are applied identically
everywhere. Videos join the same pipeline through their *poster frame*: a
frame pulled from ~1 second in, which is what gets embedded, face-scanned,
hashed, and thumbnailed — so every image feature works on videos unchanged.
"""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
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
# AVIF decodes through Pillow >= 11.2 wheels; gated on a runtime check so an
# older Pillow skips the files instead of failing every one of them.
AVIF_EXTS = {".avif"}
# Videos need the ffmpeg binary bundled with imageio-ffmpeg; without it they
# are skipped with a single warning, the same deal as pillow-heif.
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp",
              ".mpg", ".mpeg", ".wmv", ".mts", ".m2ts"}
SUPPORTED_EXTS = RASTER_EXTS | HEIF_EXTS | RAW_EXTS

# Long edge for decoded poster frames. Everything downstream (embedding,
# faces, thumbnails) wants 1600px or less, so decoding 4K pixels is waste.
POSTER_MAX_SIDE = 1920

# Keep ffmpeg from opening a console window inside the packaged GUI app.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

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


@lru_cache(maxsize=1)
def avif_supported() -> bool:
    try:
        from PIL import features

        return bool(features.check("avif"))
    except Exception:  # pragma: no cover - version dependent
        return False


@lru_cache(maxsize=1)
def ffmpeg_exe() -> Optional[str]:
    """Path to the bundled ffmpeg, or None when videos can't be handled."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("imageio-ffmpeg unavailable, videos will be skipped: %s", exc)
        return None


def video_supported() -> bool:
    return ffmpeg_exe() is not None


def is_video(path: os.PathLike | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_supported(path: os.PathLike | str) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_EXTS:
        return True
    if suffix in AVIF_EXTS:
        return avif_supported()
    if suffix in VIDEO_EXTS:
        return video_supported()
    return False


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


# -- video probing and poster frames ------------------------------------

@dataclass(frozen=True)
class VideoInfo:
    duration_ms: int
    width: int          # display dimensions, rotation already applied
    height: int
    taken_at: Optional[datetime]


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_STREAM_RE = re.compile(r"Stream .*Video:.*?(\d{2,5})x(\d{2,5})")
_ROTATION_RE = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees|rotate\s*:\s*(-?\d+)")
_CREATION_RE = re.compile(r"creation_time\s*:\s*(\S+)")


def _run_ffmpeg(args: list, timeout: float = 60.0) -> subprocess.CompletedProcess:
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg is not available")
    return subprocess.run([exe, *args], capture_output=True, timeout=timeout,
                          creationflags=_NO_WINDOW)


@lru_cache(maxsize=512)
def _probe_cached(path: str, mtime_ns: int) -> VideoInfo:
    # `ffmpeg -i` with no output exits non-zero by design; the stream
    # banner on stderr is all we want.
    proc = _run_ffmpeg(["-i", path], timeout=30.0)
    text = proc.stderr.decode("utf-8", errors="replace")

    duration_ms = 0
    m = _DURATION_RE.search(text)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration_ms = int(round((h * 3600 + mi * 60 + s) * 1000))

    width = height = 0
    m = _VIDEO_STREAM_RE.search(text)
    if m:
        width, height = int(m.group(1)), int(m.group(2))

    m = _ROTATION_RE.search(text)
    if m:
        angle = abs(int(float(m.group(1) or m.group(2)))) % 360
        if angle in (90, 270):
            width, height = height, width

    taken_at = None
    m = _CREATION_RE.search(text)
    if m:
        raw = m.group(1)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            # Stored as UTC; photos carry local wall-clock times, so convert.
            taken_at = parsed.astimezone().replace(tzinfo=None)
        except ValueError:
            logger.debug("Unparseable creation_time %r in %s", raw, path)

    return VideoInfo(duration_ms=duration_ms, width=width, height=height,
                     taken_at=taken_at)


def probe_video(path: os.PathLike | str) -> VideoInfo:
    """Duration, display size, and capture time — no frame decode."""
    p = Path(path)
    return _probe_cached(str(p), p.stat().st_mtime_ns)


@lru_cache(maxsize=8)
def _poster_cached(path: str, mtime_ns: int) -> Image.Image:
    """Decode one representative frame as a PIL image.

    PNG-over-pipe rather than rawvideo: the PNG carries its own dimensions,
    so ffmpeg's automatic display-rotation can never scramble a reshape.
    Cached because one indexing pass reads the poster several times (phash,
    embedding buffer, thumbnails); callers get copies via _video_poster.
    """
    info = _probe_cached(path, mtime_ns)
    seek = min(1.0, info.duration_ms / 2000.0) if info.duration_ms else 0.0

    scale_args: list = []
    if info.width and info.height and max(info.width, info.height) > POSTER_MAX_SIDE:
        factor = POSTER_MAX_SIDE / max(info.width, info.height)
        tw = max(2, int(info.width * factor)) // 2 * 2
        th = max(2, int(info.height * factor)) // 2 * 2
        scale_args = ["-vf", f"scale={tw}:{th}"]

    for attempt_seek in ([seek, 0.0] if seek > 0 else [0.0]):
        args = ["-ss", f"{attempt_seek:.3f}", "-i", path, "-frames:v", "1",
                *scale_args, "-f", "image2pipe", "-c:v", "png", "-"]
        proc = _run_ffmpeg(args)
        if proc.returncode == 0 and proc.stdout:
            img = Image.open(io.BytesIO(proc.stdout))
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
    raise ValueError(
        f"Could not decode a frame from {path}: "
        f"{proc.stderr.decode('utf-8', errors='replace')[-300:]}")


def _video_poster(path: os.PathLike | str) -> Image.Image:
    p = Path(path)
    return _poster_cached(str(p), p.stat().st_mtime_ns).copy()


def open_image(path: os.PathLike | str, target: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Open an image — or a video's poster frame — as RGB, oriented.

    ``target`` enables JPEG draft mode: libjpeg decodes straight to a reduced
    resolution, which is several times faster and uses a fraction of the
    memory when all we need is a thumbnail or a 224px model input.
    """
    if is_video(path):
        return _video_poster(path)
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
    if is_video(path):
        info = probe_video(path)
        if info.width and info.height:
            return info.width, info.height
        with _video_poster(path) as poster:
            return poster.size
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
