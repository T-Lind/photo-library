"""Image/text embedding backends.

The backend is chosen with ``PHOTO_EMBED_BACKEND`` and is swappable without
touching any other module — the rest of the system only ever sees a
normalised ``(n, dim)`` float32 array.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..config import Settings, get_settings
from .base import Embedder, ImageInput, l2_normalize

_instance: Optional[Embedder] = None
_lock = threading.Lock()


def build_embedder(settings: Optional[Settings] = None) -> Embedder:
    """Construct (but do not load) the configured embedder."""
    s = settings or get_settings()

    if s.embed_backend == "stub":
        from .stub import StubEmbedder

        return StubEmbedder(model_name=s.embed_model if s.embed_model.startswith("stub") else "stub-v1")

    if s.embed_backend == "onnx":
        from .onnx_vision import OnnxVisionEmbedder

        return OnnxVisionEmbedder(
            s.onnx_model_dir, batch_size=s.embed_batch_size,
            prefer_int8=s.onnx_int8, threads=s.onnx_threads)

    if s.embed_backend == "open_clip":
        from .hf_vision import OpenClipEmbedder

        return OpenClipEmbedder(
            model_name=s.embed_model, device=s.device,
            fp16=s.embed_fp16, batch_size=s.embed_batch_size)

    from .hf_vision import DEFAULTS, HFVisionEmbedder

    model = s.embed_model
    # Guard against a config that names a backend but leaves the other
    # family's default model in place.
    if s.embed_backend == "clip" and "siglip" in model:
        model = DEFAULTS["clip"]
    if s.embed_backend == "siglip" and "clip-vit" in model:
        model = DEFAULTS["siglip"]

    return HFVisionEmbedder(
        backend=s.embed_backend, model_name=model, device=s.device,
        fp16=s.embed_fp16, batch_size=s.embed_batch_size)


def get_embedder(settings: Optional[Settings] = None) -> Embedder:
    """Process-wide singleton — model weights are loaded at most once."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = build_embedder(settings)
    return _instance


def set_embedder(embedder: Optional[Embedder]) -> None:
    """Override the singleton (tests, or an explicit CLI-provided model)."""
    global _instance
    with _lock:
        _instance = embedder


__all__ = ["Embedder", "ImageInput", "l2_normalize", "build_embedder",
           "get_embedder", "set_embedder"]
