"""ONNX Runtime embedder — the backend the packaged desktop app uses.

PyTorch is roughly 2.5 GB installed and is the single most awkward
dependency to freeze into a standalone binary. onnxruntime is about 50 MB,
is already required for face recognition, and runs the same graph. Exporting
the model once (see ``tools/export_onnx.py``) and shipping the ONNX files
turns a ~3 GB installer into a ~500 MB one and removes the whole PyTorch
packaging problem.

Preprocessing is reproduced here in NumPy from the parameters the exporter
read off the real 🤗 processor, so it cannot silently drift from what the
model was trained with. ``golden.json`` — written by the exporter — lets the
runtime prove that byte for byte; see :meth:`OnnxVisionEmbedder.self_check`.
"""

from __future__ import annotations

import json
import logging
import re
import string
import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from ..imageio import open_image
from .base import Embedder, ImageLike, as_image_inputs, l2_normalize

logger = logging.getLogger(__name__)

# PIL resample ids, as stored by the exporter.
RESAMPLE = {
    0: Image.Resampling.NEAREST,
    1: Image.Resampling.LANCZOS,
    2: Image.Resampling.BILINEAR,
    3: Image.Resampling.BICUBIC,
    4: Image.Resampling.BOX,
    5: Image.Resampling.HAMMING,
}

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE = re.compile(r"\s+")


def canonicalize_text(text: str) -> str:
    """SigLIP's text canonicalisation: lowercase, drop punctuation, squeeze space.

    Applied only when the exporter observed the real processor doing it.
    """
    return _WHITESPACE.sub(" ", _PUNCT.sub("", text.lower())).strip()


class ModelDirError(RuntimeError):
    pass


class OnnxVisionEmbedder(Embedder):
    backend = "onnx"

    def __init__(self, model_dir: str | Path, providers: Optional[List[str]] = None,
                 batch_size: int = 16, prefer_int8: bool = False,
                 threads: int = 0):
        self.dir = Path(model_dir)
        if not (self.dir / "preprocess.json").exists():
            raise ModelDirError(
                f"{self.dir} is not an exported model directory "
                "(no preprocess.json). Run tools/export_onnx.py, or point "
                "PHOTO_ONNX_MODEL_DIR at one.")

        self.config = json.loads((self.dir / "preprocess.json").read_text())
        self.model_name = self.config.get("model", str(self.dir.name))
        self.image_cfg = self.config["image"]
        self.text_cfg = self.config["text"]
        self._dim = int(self.config["dim"])
        self.batch_size = batch_size
        self.prefer_int8 = prefer_int8
        self.threads = threads
        self._providers = providers

        self._vision = None
        self._text = None
        self._tokenizer = None
        self._lock = threading.Lock()

        mean = np.asarray(self.image_cfg["image_mean"], dtype=np.float32)
        std = np.asarray(self.image_cfg["image_std"], dtype=np.float32)
        self._mean = mean.reshape(1, 1, -1)
        self._std = np.maximum(std.reshape(1, 1, -1), 1e-8)

    # -- session management ---------------------------------------------
    def _providers_list(self) -> List[str]:
        if self._providers:
            return self._providers
        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
        except Exception:  # pragma: no cover
            return ["CPUExecutionProvider"]
        preferred = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider")
                     if p in available]
        return preferred + ["CPUExecutionProvider"]

    def _pick(self, name: str) -> Path:
        int8 = self.dir / f"{name}.int8.onnx"
        if self.prefer_int8 and int8.exists():
            return int8
        path = self.dir / f"{name}.onnx"
        if not path.exists():
            raise ModelDirError(f"Missing {path}")
        return path

    def _session(self, name: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.threads:
            options.intra_op_num_threads = self.threads
        return ort.InferenceSession(str(self._pick(name)), options,
                                    providers=self._providers_list())

    @property
    def vision(self):
        if self._vision is None:
            with self._lock:
                if self._vision is None:
                    logger.info("Loading ONNX vision tower from %s", self.dir)
                    self._vision = self._session("vision")
        return self._vision

    @property
    def text(self):
        if self._text is None:
            with self._lock:
                if self._text is None:
                    self._text = self._session("text")
        return self._text

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    from tokenizers import Tokenizer

                    tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
                    length = self.text_cfg["max_length"]
                    # The text tower is exported at a fixed sequence length,
                    # so every input must be padded to exactly that.
                    tok.enable_truncation(max_length=length)
                    tok.enable_padding(length=length,
                                       pad_id=self.text_cfg.get("pad_token_id", 0),
                                       pad_token=self.text_cfg.get("pad_token", "<pad>"))
                    self._tokenizer = tok
        return self._tokenizer

    @property
    def dim(self) -> int:
        return self._dim

    # -- preprocessing ---------------------------------------------------
    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Reproduce the 🤗 image processor exactly, in NumPy. Returns CHW."""
        cfg = self.image_cfg
        resample = RESAMPLE.get(int(cfg.get("resample", 3)), Image.Resampling.BICUBIC)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if cfg.get("do_center_crop"):
            # CLIP-style: scale the short edge, then crop the centre.
            target = min(cfg["height"], cfg["width"])
            scale = target / min(img.size)
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                resample)
            crop_w, crop_h = cfg["crop_width"], cfg["crop_height"]
            left = (img.width - crop_w) // 2
            top = (img.height - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
        else:
            # SigLIP-style: squash straight to the target square.
            img = img.resize((cfg["width"], cfg["height"]), resample)

        array = np.asarray(img, dtype=np.float32) * float(cfg["rescale_factor"])
        array = (array - self._mean) / self._std
        return np.transpose(array, (2, 0, 1))

    def preprocess_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.text_cfg.get("canonicalize"):
            texts = [canonicalize_text(t) for t in texts]
        encoded = self.tokenizer.encode_batch(list(texts))
        return np.asarray([e.ids for e in encoded], dtype=np.int64)

    # -- inference -------------------------------------------------------
    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        items = as_image_inputs(images)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: List[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            tensors = []
            for item in chunk:
                if item.array is not None:
                    tensors.append(self.preprocess_image(Image.fromarray(item.array)))
                else:
                    with open_image(item.path, target=(384, 384)) as img:
                        tensors.append(self.preprocess_image(img))
            batch = np.stack(tensors).astype(np.float32)
            out.append(self.vision.run(None, {"pixel_values": batch})[0])
        return l2_normalize(np.concatenate(out, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        ids = self.preprocess_texts(texts)
        return l2_normalize(self.text.run(None, {"input_ids": ids})[0])

    # -- verification ----------------------------------------------------
    def self_check(self, tolerance: float = 1e-3) -> dict:
        """Compare against the golden vectors the exporter recorded.

        Preprocessing reimplemented in a second language is exactly the kind
        of thing that goes subtly wrong and shows up as "search got worse"
        months later. This turns that into a hard, checkable failure.
        """
        golden_path = self.dir / "golden.json"
        if not golden_path.exists():
            return {"checked": False,
                    "reason": "no golden.json in the model directory"}

        golden = json.loads(golden_path.read_text())
        report: dict = {"checked": True, "model": self.model_name}

        expected_ids = golden.get("token_ids")
        if expected_ids:
            actual = self.preprocess_texts(golden["texts"]).tolist()
            report["tokenizer_matches"] = actual == expected_ids
            if not report["tokenizer_matches"]:
                report["tokenizer_expected"] = expected_ids[0][:16]
                report["tokenizer_actual"] = actual[0][:16]

        expected_text = golden.get("text_embeddings")
        if expected_text:
            actual = self.embed_texts(golden["texts"])
            expected = l2_normalize(np.asarray(expected_text, dtype=np.float32))
            worst = float((actual * expected).sum(-1).min())
            report["text_cosine"] = round(worst, 6)
            report["text_matches"] = (1.0 - worst) <= tolerance

        expected_image = golden.get("image_embeddings")
        if expected_image:
            pixels = np.asarray(golden["pixels"], dtype=np.float32)
            actual = l2_normalize(self.vision.run(None, {"pixel_values": pixels})[0])
            expected = l2_normalize(np.asarray(expected_image, dtype=np.float32))
            worst = float((actual * expected).sum(-1).min())
            report["image_cosine"] = round(worst, 6)
            report["image_matches"] = (1.0 - worst) <= tolerance

        report["ok"] = all(v for k, v in report.items() if k.endswith("_matches"))
        return report
