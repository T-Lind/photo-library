"""Face detection, recognition, and identity clustering."""

from __future__ import annotations

import threading
from typing import Optional

from ..config import Settings, get_settings
from .base import DetectedFace, FaceBackend, score_quality

_instance: Optional[FaceBackend] = None
_lock = threading.Lock()


class NullFaceBackend(FaceBackend):
    """Disables face processing entirely (``PHOTO_FACE_BACKEND=none``)."""

    backend = "none"
    model_name = "none"

    @property
    def dim(self) -> int:
        return 512

    def detect(self, image):  # noqa: D102
        return []


def build_face_backend(settings: Optional[Settings] = None) -> FaceBackend:
    s = settings or get_settings()

    if s.face_backend == "none":
        return NullFaceBackend()
    if s.face_backend == "stub":
        from .stub import StubFaceBackend

        return StubFaceBackend()
    if s.face_backend == "dlib":
        from .dlib_backend import DlibFaceBackend

        return DlibFaceBackend(min_face_size=s.face_min_size)

    from .insight import InsightFaceBackend

    return InsightFaceBackend(
        model_name=s.face_model,
        det_size=s.face_det_size,
        min_det_score=s.face_min_det_score,
        min_face_size=s.face_min_size,
        device=s.device,
    )


def get_face_backend(settings: Optional[Settings] = None) -> FaceBackend:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = build_face_backend(settings)
    return _instance


def set_face_backend(backend: Optional[FaceBackend]) -> None:
    global _instance
    with _lock:
        _instance = backend


__all__ = [
    "DetectedFace", "FaceBackend", "NullFaceBackend", "score_quality",
    "build_face_backend", "get_face_backend", "set_face_backend",
]
