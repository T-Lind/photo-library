"""SigLIP 2 / CLIP image-text embedders backed by 🤗 transformers.

SigLIP 2 is the default. Compared with the original OpenAI CLIP that this
project used before, it is trained with a sigmoid pairwise loss plus caption
grounding and self-distillation, and it retrieves noticeably better on the
kind of query a photo library actually gets ("kids on a beach at sunset",
"birthday cake with candles"). It is Apache-2.0 and runs fully offline.

Both families expose ``get_image_features`` / ``get_text_features``, so one
implementation covers them; only the text tokenisation differs — SigLIP was
trained with fixed-length padded sequences and degrades if you pad
dynamically.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Sequence

import numpy as np

from ..imageio import open_image
from .base import Embedder, ImageLike, as_image_inputs, l2_normalize

logger = logging.getLogger(__name__)

# Preferred model per family. Bigger variants are a drop-in swap via
# PHOTO_EMBED_MODEL when a GPU is available:
#   google/siglip2-so400m-patch14-384  (1152d, best quality)
#   google/siglip2-large-patch16-256   (1024d)
DEFAULTS = {
    "siglip": "google/siglip2-base-patch16-224",
    "clip": "openai/clip-vit-base-patch16",
}

# SigLIP's text tower is trained at a fixed 64-token context.
SIGLIP_TEXT_LEN = 64


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class HFVisionEmbedder(Embedder):
    """Lazily-loaded transformers embedder, safe to share across threads."""

    def __init__(self, backend: str = "siglip", model_name: str | None = None,
                 device: str = "auto", fp16: bool = True, batch_size: int = 16):
        self.backend = backend
        self.model_name = model_name or DEFAULTS.get(backend, DEFAULTS["siglip"])
        self.device = pick_device(device)
        self.fp16 = fp16 and self.device == "cuda"
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._dim: int | None = None
        # transformers models are not thread-safe for concurrent forward
        # passes on the same CUDA stream; the API serves requests from a
        # thread pool, so guard the call.
        self._lock = threading.Lock()

    # -- lazy loading ----------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoProcessor

            logger.info("Loading %s embedder %s on %s",
                        self.backend, self.model_name, self.device)
            dtype = torch.float16 if self.fp16 else torch.float32
            model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype)
            self._model = model.to(self.device).eval()
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._dim = int(
                getattr(model.config, "projection_dim", 0)
                or getattr(getattr(model.config, "text_config", None), "hidden_size", 0)
                or 768
            )

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return int(self._dim)

    # -- inference -------------------------------------------------------
    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        self._ensure_loaded()
        items = as_image_inputs(images)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        import torch

        out: List[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            batch, opened = [], []
            try:
                for item in chunk:
                    if item.array is not None:
                        batch.append(item.array)
                    else:
                        # Decode straight to roughly the model's input size;
                        # for JPEGs this skips most of the decode work.
                        img = open_image(item.path, target=(384, 384))
                        opened.append(img)
                        batch.append(img)
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.fp16 and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].half()
                with self._lock, torch.inference_mode():
                    feats = self._model.get_image_features(**inputs)
                out.append(feats.float().cpu().numpy())
            finally:
                for img in opened:
                    img.close()
        return l2_normalize(np.concatenate(out, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        import torch

        if self.backend == "siglip":
            kwargs = dict(padding="max_length", max_length=SIGLIP_TEXT_LEN,
                          truncation=True)
        else:
            kwargs = dict(padding=True, truncation=True)

        inputs = self._processor(text=list(texts), return_tensors="pt", **kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._lock, torch.inference_mode():
            feats = self._model.get_text_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())


class OpenClipEmbedder(Embedder):
    """open_clip backend, for LAION/MetaCLIP checkpoints.

    Model names use open_clip's ``arch:pretrained`` form, e.g.
    ``ViT-H-14-quickgelu:dfn5b``.
    """

    backend = "open_clip"

    def __init__(self, model_name: str = "ViT-B-16:laion2b_s34b_b88k",
                 device: str = "auto", fp16: bool = True, batch_size: int = 16):
        self.model_name = model_name
        self.device = pick_device(device)
        self.fp16 = fp16 and self.device == "cuda"
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import open_clip
            import torch

            arch, _, pretrained = self.model_name.partition(":")
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained or None)
            self._model = model.to(self.device).eval()
            if self.fp16:
                self._model = self._model.half()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(arch)
            with torch.inference_mode():
                probe = self._model.encode_text(self._tokenizer(["probe"]).to(self.device))
            self._dim = int(probe.shape[-1])

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return int(self._dim)

    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        self._ensure_loaded()
        items = as_image_inputs(images)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        from PIL import Image as PILImage
        import torch

        out = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            tensors = []
            for item in chunk:
                if item.array is not None:
                    tensors.append(self._preprocess(PILImage.fromarray(item.array)))
                    continue
                with open_image(item.path, target=(384, 384)) as img:
                    tensors.append(self._preprocess(img))
            batch = torch.stack(tensors).to(self.device)
            if self.fp16:
                batch = batch.half()
            with self._lock, torch.inference_mode():
                feats = self._model.encode_image(batch)
            out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        import torch

        tokens = self._tokenizer(list(texts)).to(self.device)
        with self._lock, torch.inference_mode():
            feats = self._model.encode_text(tokens)
        return l2_normalize(feats.float().cpu().numpy())
