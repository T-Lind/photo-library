"""Face detection/recognition interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DetectedFace:
    """One detected face and its identity embedding."""

    bbox: Tuple[int, int, int, int]        # x, y, w, h in the original image
    det_score: float                        # detector confidence, 0..1
    embedding: np.ndarray                   # L2-normalised identity vector
    quality: float = 1.0                    # 0..1, see `score_quality`
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class FaceBackend(ABC):
    """Detects faces in an image and embeds each one for recognition."""

    backend: str = "base"
    model_name: str = ""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect and embed every face in one RGB uint8 HxWx3 array."""

    def detect_batch(self, images: Sequence[np.ndarray]) -> List[List[DetectedFace]]:
        """Default: per-image. Backends with real batching should override."""
        return [self.detect(img) for img in images]

    def describe(self) -> dict:
        return {"backend": self.backend, "model": self.model_name, "dim": self.dim}


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur estimate: variance of the Laplacian.

    Sharp crops have high-frequency energy; motion-blurred or badly upscaled
    ones don't. Cheap enough to run on every face crop.
    """
    if gray.size == 0:
        return 0.0
    g = gray.astype(np.float32)
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def score_quality(image: np.ndarray, bbox: Tuple[int, int, int, int],
                  det_score: float) -> float:
    """Combine detector confidence, face size, and sharpness into 0..1.

    Quality drives two things: which face becomes a person's cover image, and
    how much weight a face carries when its person's centroid is recomputed.
    Letting a blurry 30px background face pull the centroid around is a large
    part of why naive clustering produces "one person" clusters that contain
    four different people.
    """
    x, y, w, h = bbox
    crop = image[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return 0.0

    # Size: 40px is barely usable, 200px+ is plenty.
    size_score = min(1.0, max(0.0, (min(w, h) - 40) / 160.0))

    gray = crop.mean(axis=2) if crop.ndim == 3 else crop
    sharpness = laplacian_variance(gray)
    # ~100 is the usual "in focus" threshold for 8-bit luminance.
    sharp_score = min(1.0, sharpness / 100.0)

    return float(np.clip(
        0.45 * det_score + 0.35 * size_score + 0.20 * sharp_score, 0.0, 1.0))
