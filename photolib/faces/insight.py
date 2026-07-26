"""InsightFace backend — RetinaFace detection + ArcFace recognition.

This replaces dlib/``face_recognition``, which was the weakest link in the
old pipeline. dlib's ResNet embedding is a 2017 model trained on ~3M faces;
ArcFace ``w600k_r50`` (the recogniser inside ``buffalo_l``) is trained on
WebFace600K with an angular-margin loss and is dramatically better on the
cases a family library is full of: profile views, faces at an angle, poor
light, children as they age, and people wearing glasses or hats.

Practically it also runs through ONNX Runtime, which means real batching and
optional GPU/CoreML acceleration, where dlib was single-threaded CPU.
"""

from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np

from .base import DetectedFace, FaceBackend, score_quality

logger = logging.getLogger(__name__)


def _providers(device: str) -> List[str]:
    """ONNX Runtime execution providers, best-available first."""
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - optional dependency
        return ["CPUExecutionProvider"]

    preferred = []
    if device in ("auto", "cuda"):
        preferred += ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
    if device in ("auto", "mps"):
        preferred += ["CoreMLExecutionProvider"]
    chosen = [p for p in preferred if p in available]
    chosen.append("CPUExecutionProvider")
    return chosen


class InsightFaceBackend(FaceBackend):
    backend = "insightface"

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640,
                 min_det_score: float = 0.5, min_face_size: int = 40,
                 device: str = "auto", model_root: str | None = None):
        self.model_name = model_name
        self.det_size = det_size
        self.min_det_score = min_det_score
        self.min_face_size = min_face_size
        self.device = device
        # InsightFace resolves models under <root>/models/<name>. Keeping
        # this inside the app's data directory rather than ~/.insightface
        # makes a packaged install self-contained.
        self.model_root = model_root
        self._app = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._app is not None:
            return self._app
        with self._lock:
            if self._app is not None:
                return self._app
            from insightface.app import FaceAnalysis

            logger.info("Loading InsightFace %s (det_size=%d)",
                        self.model_name, self.det_size)
            kwargs = {}
            if self.model_root:
                from pathlib import Path

                # FaceAnalysis resolves <root>/models/<name>, which is exactly
                # where photolib.models downloads to.
                kwargs["root"] = str(Path(self.model_root).expanduser())
            app = FaceAnalysis(
                name=self.model_name,
                # Detection + recognition only. The gender/age and 3D landmark
                # heads roughly double per-image cost and nothing here uses them.
                allowed_modules=["detection", "recognition"],
                providers=_providers(self.device),
            )
            app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            self._app = app
            return app

    @property
    def dim(self) -> int:
        return 512  # ArcFace w600k_r50

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        app = self._ensure_loaded()
        # InsightFace expects BGR (it is built on OpenCV conventions).
        bgr = image[:, :, ::-1]
        with self._lock:
            found = app.get(bgr)

        faces: List[DetectedFace] = []
        for f in found:
            score = float(getattr(f, "det_score", 1.0))
            if score < self.min_det_score:
                continue
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x, y = max(0, x1), max(0, y1)
            w, h = max(0, x2 - x), max(0, y2 - y)
            if min(w, h) < self.min_face_size:
                continue

            # `normed_embedding` is already unit-length; fall back defensively.
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = f.embedding / max(np.linalg.norm(f.embedding), 1e-12)

            bbox = (x, y, w, h)
            faces.append(DetectedFace(
                bbox=bbox,
                det_score=score,
                embedding=np.asarray(emb, dtype=np.float32),
                quality=score_quality(image, bbox, score),
                landmarks=getattr(f, "kps", None),
            ))
        return faces
