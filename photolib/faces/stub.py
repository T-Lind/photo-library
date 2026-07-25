"""Deterministic face backend for tests.

Encodes identity in the image itself: a synthetic test image contains one
solid-colour block per "person", and each distinct colour maps to a stable
embedding. That gives the clustering code real, checkable behaviour (same
person across images clusters together, different people don't) without any
model weights.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .base import DetectedFace, FaceBackend

MARKER_ROW_HEIGHT = 16
MARKER_WIDTH = 16


def identity_vector(identity: str, dim: int = 32) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


class StubFaceBackend(FaceBackend):
    """Reads "faces" from coloured marker blocks along the image's top row.

    Each ``MARKER_WIDTH``-wide block of non-white pixels in the top strip is
    one face; its RGB triple is the identity.
    """

    backend = "stub"
    model_name = "stub-face-v1"

    def __init__(self, dim: int = 32, noise: float = 0.0):
        self._dim = dim
        self.noise = noise

    @property
    def dim(self) -> int:
        return self._dim

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        faces: List[DetectedFace] = []
        if image.ndim != 3 or image.shape[0] < MARKER_ROW_HEIGHT:
            return faces

        strip = image[:MARKER_ROW_HEIGHT]
        width = image.shape[1]
        for i, x in enumerate(range(0, width - MARKER_WIDTH + 1, MARKER_WIDTH)):
            # Sample the block's interior: JPEG ringing at a colour boundary
            # would otherwise blur two adjacent markers into each other.
            block = strip[4:-2, x + 4:x + MARKER_WIDTH - 4]
            if block.size == 0:
                continue
            mean = block.reshape(-1, 3).mean(axis=0)
            # Near-white is background, near-black is padding — neither is a face.
            if mean.min() > 235 or mean.max() < 20:
                continue
            # Quantise so lossy compression can't change a person's identity.
            colour = tuple(int(round(c / 32.0)) for c in mean)
            emb = identity_vector(f"{colour[0]}-{colour[1]}-{colour[2]}", self._dim)
            if self.noise:
                rng = np.random.default_rng(abs(hash(colour)) % (2 ** 32) + i)
                emb = emb + rng.standard_normal(self._dim).astype(np.float32) * self.noise
                emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
            faces.append(DetectedFace(
                bbox=(x, 0, MARKER_WIDTH, MARKER_ROW_HEIGHT),
                det_score=0.99,
                embedding=emb,
                quality=0.9,
            ))
        return faces
