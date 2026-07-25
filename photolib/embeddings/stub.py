"""Deterministic offline embedder used by tests and CI.

It maps text and images into the same space by hashing *words*: an image's
"content" is taken from its filename, so ``beach-sunset-01.jpg`` genuinely
ranks first for the query "sunset on the beach". That makes it possible to
test ranking, filtering, and pagination end-to-end without downloading a
gigabyte of model weights.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .base import Embedder, ImageLike, as_image_inputs, l2_normalize

_WORD = re.compile(r"[a-z0-9]+")


def _word_vector(word: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim, dtype=np.float32)


def _bag_vector(text: str, dim: int) -> np.ndarray:
    words = _WORD.findall(text.lower())
    if not words:
        return np.ones(dim, dtype=np.float32)
    return np.sum([_word_vector(w, dim) for w in words], axis=0)


class StubEmbedder(Embedder):
    backend = "stub"

    def __init__(self, dim: int = 64, model_name: str = "stub-v1"):
        self._dim = dim
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        items = as_image_inputs(images)
        if not items:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(
            np.stack([_bag_vector(Path(i.path).stem, self._dim) for i in items]))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([_bag_vector(t, self._dim) for t in texts]))
