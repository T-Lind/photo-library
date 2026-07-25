"""Embedder interface and shared helpers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class ImageInput:
    """One image to embed, optionally with its pixels already decoded.

    The indexer decodes each photo exactly once and hands the same array to
    both the embedder and the face detector. Decoding a 12MP JPEG twice is
    the single largest avoidable cost in the whole ingest pipeline.
    """

    path: str
    array: Optional[np.ndarray] = None


ImageLike = Union[ImageInput, str, "os.PathLike[str]", Path]


def as_image_inputs(items: Sequence[ImageLike]) -> List[ImageInput]:
    return [it if isinstance(it, ImageInput) else ImageInput(path=str(it))
            for it in items]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Project rows onto the unit sphere.

    Every vector in the library is stored normalised so that cosine
    similarity is a plain dot product and the ANN index's cosine metric is
    exact rather than an approximation over unnormalised magnitudes.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (x / norms).astype(np.float32)


class Embedder(ABC):
    """Maps images and text into one shared, normalised vector space."""

    backend: str = "base"
    model_name: str = ""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        """(n, dim) float32, L2-normalised, in input order."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised, in input order."""

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0].tolist()

    def describe(self) -> dict:
        return {"backend": self.backend, "model": self.model_name, "dim": self.dim}
