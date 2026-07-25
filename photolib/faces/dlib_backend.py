"""dlib / ``face_recognition`` backend.

Kept as a fallback for machines where ONNX Runtime won't install. It is
noticeably weaker than the InsightFace backend — 128-d embeddings, frontal
bias, CPU-only — so it is not the default. Its embeddings are normalised
here so that the rest of the pipeline can treat every backend identically
and use cosine similarity throughout.
"""

from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np

from .base import DetectedFace, FaceBackend, score_quality

logger = logging.getLogger(__name__)


class DlibFaceBackend(FaceBackend):
    backend = "dlib"

    def __init__(self, model_name: str = "hog", min_face_size: int = 40,
                 upsample: int = 1, num_jitters: int = 1):
        # "hog" is fast and CPU-friendly; "cnn" is much more accurate but needs
        # a GPU to be practical.
        self.model_name = model_name
        self.min_face_size = min_face_size
        self.upsample = upsample
        self.num_jitters = num_jitters
        self._lock = threading.Lock()
        self._fr = None

    def _ensure_loaded(self):
        if self._fr is None:
            import face_recognition

            self._fr = face_recognition
        return self._fr

    @property
    def dim(self) -> int:
        return 128

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        fr = self._ensure_loaded()
        with self._lock:
            locations = fr.face_locations(
                image, number_of_times_to_upsample=self.upsample,
                model=self.model_name)
            encodings = fr.face_encodings(
                image, locations, num_jitters=self.num_jitters)

        faces: List[DetectedFace] = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            x, y = max(0, left), max(0, top)
            w, h = max(0, right - x), max(0, bottom - y)
            if min(w, h) < self.min_face_size:
                continue
            emb = np.asarray(enc, dtype=np.float32)
            emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
            bbox = (x, y, w, h)
            faces.append(DetectedFace(
                bbox=bbox,
                det_score=1.0,  # dlib's HOG detector reports no confidence
                embedding=emb,
                quality=score_quality(image, bbox, 0.9),
            ))
        return faces
