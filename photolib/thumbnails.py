"""On-disk thumbnail and face-crop cache.

Three details that matter at 200k photos:

* **Sharded directories.** Cache files live in ``<size>/<ab>/<id>.webp``.
  A single directory holding 200k entries is slow to stat on ext4 and
  painful on exFAT/NTFS external drives, which is where photo libraries
  usually live.
* **WebP.** ~30% smaller than JPEG at the same visual quality, so a grid of
  200 thumbnails is a third less data over the wire. JPEG stays available
  for anything that needs it.
* **Draft-mode decoding.** Building a 200px thumbnail from a 48MP JPEG
  decodes at 1/8 scale instead of decoding 48 million pixels and throwing
  most of them away.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .imageio import open_image

logger = logging.getLogger(__name__)

# Long-edge sizes. `grid` is what the photo wall requests; `preview` is what
# the lightbox shows before the original finishes loading.
SIZES: Dict[str, int] = {
    "small": 160,
    "medium": 320,
    "grid": 512,
    "large": 1024,
    "preview": 1600,
}

DEFAULT_QUALITY = {"webp": 82, "jpeg": 85}
FACE_CROP_SIZE = 256
FACE_CROP_PADDING = 0.35  # fraction of the box added on each side


class ThumbnailCache:
    def __init__(self, root: os.PathLike | str, fmt: str = "webp"):
        self.root = Path(root)
        self.fmt = fmt if fmt in ("webp", "jpeg") else "webp"

    # -- paths -----------------------------------------------------------
    def _path(self, kind: str, key: int, size_name: str, fmt: str) -> Path:
        shard = f"{key % 256:02x}"
        ext = "webp" if fmt == "webp" else "jpg"
        return self.root / kind / size_name / shard / f"{key}.{ext}"

    def thumbnail_path(self, image_id: int, size_name: str,
                       fmt: Optional[str] = None) -> Path:
        return self._path("thumbs", image_id, size_name, fmt or self.fmt)

    def face_path(self, face_id: int) -> Path:
        return self._path("faces", face_id, "crop", "jpeg")

    # -- generation ------------------------------------------------------
    def get_thumbnail(self, image_id: int, source: os.PathLike | str,
                      size_name: str = "grid", fmt: Optional[str] = None) -> Path:
        """Return a cached thumbnail, generating it if needed."""
        if size_name not in SIZES:
            raise KeyError(f"Unknown thumbnail size {size_name!r}")
        fmt = fmt or self.fmt
        target = self.thumbnail_path(image_id, size_name, fmt)
        if self._fresh(target, source):
            return target

        edge = SIZES[size_name]
        with open_image(source, target=(edge, edge)) as img:
            img.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            self._save(img, target, fmt)
        return target

    def get_face_crop(self, face_id: int, source: os.PathLike | str,
                      bbox: Tuple[int, int, int, int]) -> Path:
        """Crop one face out of its original photo, cached.

        Crops are generated on demand rather than written during indexing:
        a library with a million faces would otherwise spend several
        gigabytes on thumbnails most people will never look at.
        """
        target = self.face_path(face_id)
        if self._fresh(target, source):
            return target

        x, y, w, h = bbox
        pad_x, pad_y = int(w * FACE_CROP_PADDING), int(h * FACE_CROP_PADDING)
        with open_image(source) as img:
            left = max(0, x - pad_x)
            top = max(0, y - pad_y)
            right = min(img.width, x + w + pad_x)
            bottom = min(img.height, y + h + pad_y)
            if right <= left or bottom <= top:
                left, top, right, bottom = 0, 0, img.width, img.height
            crop = img.crop((left, top, right, bottom))
            crop.thumbnail((FACE_CROP_SIZE, FACE_CROP_SIZE), Image.Resampling.LANCZOS)
            self._save(crop, target, "jpeg")
        return target

    # -- internals -------------------------------------------------------
    @staticmethod
    def _fresh(target: Path, source: os.PathLike | str) -> bool:
        try:
            if not target.exists():
                return False
            return target.stat().st_mtime >= os.path.getmtime(source)
        except OSError:
            return False

    def _save(self, img: Image.Image, target: Path, fmt: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Write to a temp file in the same directory and rename: a reader
        # racing a writer sees either the old file or the new one, never a
        # half-written image.
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            if fmt == "webp":
                img.save(tmp, "WEBP", quality=DEFAULT_QUALITY["webp"], method=4)
            else:
                img.save(tmp, "JPEG", quality=DEFAULT_QUALITY["jpeg"],
                         optimize=True, progressive=True)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def pregenerate(self, image_id: int, source: os.PathLike | str,
                    sizes=("small", "grid")) -> None:
        """Warm the cache during indexing so first browse is instant."""
        for size_name in sizes:
            try:
                self.get_thumbnail(image_id, source, size_name)
            except Exception as exc:
                logger.debug("Thumbnail pregeneration failed for %s: %s", source, exc)

    def pregenerate_from_array(self, image_id: int, source: os.PathLike | str,
                               array: np.ndarray,
                               sizes=("small", "grid")) -> None:
        """Warm thumbnails from the indexer's already-decoded RGB buffer."""
        image = Image.fromarray(array)
        try:
            for size_name in sizes:
                try:
                    target = self.thumbnail_path(image_id, size_name, self.fmt)
                    if self._fresh(target, source):
                        continue
                    edge = SIZES[size_name]
                    thumb = image.copy()
                    try:
                        thumb.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                        self._save(thumb, target, self.fmt)
                    finally:
                        thumb.close()
                except Exception as exc:
                    logger.debug("Thumbnail pregeneration failed for %s: %s",
                                 source, exc)
        finally:
            image.close()

    def purge_image(self, image_id: int) -> int:
        removed = 0
        for size_name in SIZES:
            for fmt in ("webp", "jpeg"):
                p = self.thumbnail_path(image_id, size_name, fmt)
                if p.exists():
                    p.unlink()
                    removed += 1
        return removed
