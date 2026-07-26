"""Local text recognition (OCR) for screenshot-heavy libraries.

Semantic embeddings are the wrong tool for finding the literal string
"JROTC" rendered inside a screenshot: the base image model sees photos at
224px, where UI text is illegible. OCR closes that gap — text is extracted
once at index time, stored next to the image row, and exact matches rank
ahead of semantic ones at search time.

The default engine is RapidOCR, which runs PaddleOCR's detection and
recognition models on onnxruntime — the same inference stack the rest of the
application already ships. Its models are bundled inside the Python package,
so the offline guarantee holds: nothing is downloaded at runtime.

OCR is strictly optional. When the package is not installed the library
behaves exactly as before, and the API reports the engine as unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class OcrBackend(Protocol):
    name: str
    model_name: str

    def extract(self, array: np.ndarray, path: Optional[str] = None) -> str:
        """Text found in the image, reading-order lines joined by newlines."""
        ...


class RapidOcrBackend:
    """PaddleOCR det+rec models via onnxruntime (the rapidocr package)."""

    name = "rapidocr"
    model_name = "ppocr-v4-onnx"

    def __init__(self, min_confidence: float = 0.5, max_chars: int = 4000,
                 max_side: int = 1280):
        from rapidocr_onnxruntime import RapidOCR  # noqa: import guarded by build_ocr

        self._engine = RapidOCR()
        self.min_confidence = min_confidence
        self.max_chars = max_chars
        self.max_side = max_side

    def extract(self, array: np.ndarray, path: Optional[str] = None) -> str:
        img = self._shrink(array)
        result, _ = self._engine(img)
        if not result:
            return ""
        lines: List[str] = []
        for entry in result:
            # Each entry is [box, text, confidence].
            text = str(entry[1]).strip()
            confidence = float(entry[2]) if len(entry) > 2 else 1.0
            if text and confidence >= self.min_confidence:
                lines.append(text)
        return "\n".join(lines)[: self.max_chars]

    def _shrink(self, array: np.ndarray) -> np.ndarray:
        """Bound the long edge — detection cost is quadratic in resolution."""
        h, w = array.shape[:2]
        long_edge = max(h, w)
        if long_edge <= self.max_side:
            return array
        scale = self.max_side / long_edge
        try:
            import cv2

            return cv2.resize(array, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        except Exception:
            step = max(1, int(round(1 / scale)))
            return array[::step, ::step]


class StubOcrBackend:
    """Deterministic OCR for tests: the filename's words are the "text".

    Mirrors the stub embedder's convention, so a synthetic photo named
    ``beach-sunset-holiday-20180704.jpg`` "contains" the text
    ``beach sunset holiday 20180704`` without any model in CI.
    """

    name = "stub"
    model_name = "stub-ocr-v1"

    def extract(self, array: np.ndarray, path: Optional[str] = None) -> str:
        if not path:
            return ""
        stem = Path(path).stem
        return " ".join(part for part in stem.replace("_", "-").split("-") if part)


def build_ocr(settings=None) -> Optional[OcrBackend]:
    """The configured OCR engine, or None when text recognition is off.

    ``auto`` quietly degrades to None when rapidocr is not installed —
    OCR is an enhancement, never a requirement.
    """
    from .config import get_settings

    settings = settings or get_settings()
    backend = settings.ocr_backend

    if backend == "off":
        return None
    if backend == "stub":
        return StubOcrBackend()
    try:
        return RapidOcrBackend(
            min_confidence=settings.ocr_min_confidence,
            max_chars=settings.ocr_max_chars,
            max_side=settings.ocr_max_side,
        )
    except ImportError:
        if backend == "rapidocr":
            raise RuntimeError(
                "PHOTO_OCR_BACKEND=rapidocr but the rapidocr-onnxruntime "
                "package is not installed. `pip install rapidocr-onnxruntime` "
                "or set PHOTO_OCR_BACKEND=off.")
        logger.info("OCR disabled: rapidocr-onnxruntime is not installed")
        return None
    except Exception as exc:
        if backend == "rapidocr":
            raise
        logger.warning("OCR disabled: engine failed to initialise: %s", exc)
        return None
